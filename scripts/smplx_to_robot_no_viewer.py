import argparse
import pathlib
import os
import pickle

import numpy as np
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

from rich import print

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        help="Base path to save the robot motion (will save two .pkl files: _v1.pkl and _v2.pkl).",
    )
    
    parser.add_argument(
        "--height_adjust",
        default=False,
        action="store_true",
        help="Adjust height to ensure lowest part is on the ground.",
    )
    
    parser.add_argument(
        "--root_origin_offset",
        default=False,
        action="store_true",
        help="Offset root position using the first frame.",
    )

    parser.add_argument(
        "--contact_filter",
        default=False,
        help="Filter contacts to ensure the robot feet on ground.",
        type=bool,
    )

    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    # Load SMPLX trajectory
    print(f"Loading SMPLX file: {args.smplx_file}")
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    print(f"Loaded {len(smplx_data_frames)} frames at {aligned_fps} fps")
   
    # Initialize the retargeting system
    print(f"Initializing retarget for robot: {args.robot}")
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        aligned_fps=aligned_fps,
        contact_filter=args.contact_filter,
    )
    
    # Retarget all frames
    print("Retargeting motion...")
    qpos_list = []
    tracking_links_pos_list = []
    foot_contact_list = []
    
    for smplx_frame_data in smplx_data_frames:
        retarget.update_targets(smplx_frame_data)
        qpos = retarget.retarget()
        tracking_links_pos = retarget.get_human_tracking_targets()
        foot_contact_list.append(np.asarray(retarget.last_foot_contact, dtype=np.float32).copy())
        qpos_list.append(qpos)
        tracking_links_pos_list.append(tracking_links_pos)
    qpos_list = np.array(qpos_list)
    foot_contact_array = np.array(foot_contact_list)  # Shape: (num_frames, 2)
    print(f"Retargeted {len(qpos_list)} frames")
    

    root_pos = np.array([qpos[:3] for qpos in qpos_list])   
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    num_frames = root_pos.shape[0]

    # Initialize kinematics model
    device = "cpu"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)
    
    # Compute local body positions (with zero root pos/rot)
    print("Computing forward kinematics for local body positions...")
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )
    
    body_names = kinematics_model.body_names
    
    # Compute global body positions and rotations (with actual root pos/rot)
    print("Computing forward kinematics for global body positions...")
    body_pos, body_rot = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )  # Shape: TxNx3 for body_pos, TxNx4 for body_rot
    
    # height adjust to ensure the lowerset part is on the ground
    ground_offset = 0.0
    lowest_height = None
    if args.height_adjust:
        lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
    
    # offset using the first frame
    if args.root_origin_offset:
        root_pos[:, :2] -= root_pos[0, :2]
    
    motion_data_v1 = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
        "foot_contact": foot_contact_array,
    }
    
    # height adjust to ensure the lowerset part is on the ground
    if args.height_adjust:
        lowest_z = np.inf
        for frame in tracking_links_pos_list:
            for _, (pos, _) in frame.items():
                z = float(pos[2])
                if z < lowest_z:
                    lowest_z = z

        z_shift = lowest_z - ground_offset  # subtract this from every target z

        for frame in tracking_links_pos_list:
            for k, (pos, quat) in frame.items():
                pos2 = np.asarray(pos).copy()
                pos2[2] -= z_shift
                frame[k] = (pos2, quat)

    # offset using the first frame
    if args.root_origin_offset:
        xy_offset = np.asarray(tracking_links_pos_list[0]["pelvis"][0], dtype=float)[:2].copy()
    else:
        xy_offset = np.zeros(2, dtype=float)

    # Apply xy_offset to all tracking links (including pelvis)
    for frame in tracking_links_pos_list:
        for k, (pos, quat) in frame.items():
            pos2 = np.asarray(pos, dtype=float).copy()
            pos2[:2] -= xy_offset
            frame[k] = (pos2, quat)
    
    # Map robot link names to OptiTrack names
    ROBOT_TO_OPTITRACK = {
        "left_ankle_roll_link": "LeftFoot",
        "right_ankle_roll_link": "RightFoot",
        "left_wrist_yaw_link": "LeftHand",
        "right_wrist_yaw_link": "RightHand",
        "torso_link": "Spine1",
        "pelvis": "Hips",
    }
    
    # Collect all unique robot link names from all frames
    all_robot_link_names = set()
    for frame in tracking_links_pos_list:
        all_robot_link_names.update(frame.keys())
    
    # Build link_names list in the order of ROBOT_TO_OPTITRACK
    link_names = []
    # Add links in the order defined by ROBOT_TO_OPTITRACK
    for robot_link_name, optitrack_name in ROBOT_TO_OPTITRACK.items():
        if robot_link_name in all_robot_link_names:
            link_names.append(optitrack_name)
    
    # Add any remaining robot link names not in ROBOT_TO_OPTITRACK (keep as-is)
    remaining_links = all_robot_link_names - set(ROBOT_TO_OPTITRACK.keys())
    for robot_link_name in sorted(remaining_links):
        link_names.append(robot_link_name)
    num_links = len(link_names)
    num_frames = len(tracking_links_pos_list)
    
    # Create reverse mapping for lookup
    OPTITRACK_TO_ROBOT = {v: k for k, v in ROBOT_TO_OPTITRACK.items()}
    
    # Create arrays for pos and quat: (num_frames, num_links, 3/4)
    pos_array = np.zeros((num_frames, num_links, 3), dtype=np.float32)
    quat_array = np.zeros((num_frames, num_links, 4), dtype=np.float32)
    frame_id_array = np.arange(num_frames, dtype=np.int32)
    
    # Fill in the data from tracking_links_pos_list
    # link_names contains OptiTrack names, but we need robot link names to access the data
    for frame_idx in range(num_frames):
        frame_data = tracking_links_pos_list[frame_idx]
        for link_idx, optitrack_name in enumerate(link_names):
            # Map OptiTrack name back to robot link name
            if optitrack_name in OPTITRACK_TO_ROBOT:
                robot_link_name = OPTITRACK_TO_ROBOT[optitrack_name]
            else:
                robot_link_name = optitrack_name
            
            if robot_link_name in frame_data:
                pos, quat = frame_data[robot_link_name]
                pos_array[frame_idx, link_idx] = np.asarray(pos, dtype=np.float32)
                quat_array[frame_idx, link_idx] = np.asarray(quat, dtype=np.float32)
    
    # Convert numpy arrays to torch Tensors
    pos_tensor = torch.from_numpy(pos_array).to(dtype=torch.float32)
    quat_tensor = torch.from_numpy(quat_array).to(dtype=torch.float32)
    frame_id_tensor = torch.from_numpy(frame_id_array).to(dtype=torch.int32)
    foot_contact_tensor = torch.from_numpy(foot_contact_array).to(dtype=torch.float32)

    motion_data_v2 = {
        "fps": aligned_fps,
        "link_names": tuple(link_names),  # Convert to tuple to match expected format
        "pos": pos_tensor,  # Shape: (num_frames, num_links, 3)
        "quat": quat_tensor,  # Shape: (num_frames, num_links, 4)
        "frame_id": frame_id_tensor,  # Shape: (num_frames,)
        "foot_contact": foot_contact_tensor,  # Shape: (num_frames, 2) - foot contact for left and right feet
    }

    if args.save_path is not None:
        # Prepare file paths for both versions
        base_path = args.save_path
        if base_path.endswith('.pkl'):
            base_path = base_path[:-4]  # Remove .pkl extension
        
        save_path_v1 = base_path + '_v1.pkl'
        save_path_v2 = base_path + '_v2.pkl'
        
        save_dir = os.path.dirname(save_path_v1)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        
        # Save version 1 (simple, matching smplx_to_robot.py)
        print(f"Saving version 1 to {save_path_v1}...")
        with open(save_path_v1, "wb") as f:
            pickle.dump(motion_data_v1, f)
        
        # Save version 2 (with body_pos)
        print(f"Saving version 2 to {save_path_v2}...")
        with open(save_path_v2, "wb") as f:
            pickle.dump(motion_data_v2, f)
        
