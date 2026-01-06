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
        default=True,
        action="store_true",
        help="Adjust height to ensure lowest part is on the ground.",
    )
    
    parser.add_argument(
        "--root_origin_offset",
        default=True,
        action="store_true",
        help="Offset root position using the first frame.",
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
    print(f"Initializing retargeter for robot: {args.robot}")
    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        aligned_fps=aligned_fps,
    )
    
    # Retarget all frames
    print("Retargeting motion...")
    qpos_list = []
    for smplx_frame_data in smplx_data_frames:
        qpos, _ = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())
    
    qpos_list = np.array(qpos_list)
    print(f"Retargeted {len(qpos_list)} frames")
    
    # Extract root_pos, root_rot, and dof_pos from qpos
    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7]
    # Convert from wxyz to xyzw
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]
    
    # Initialize kinematics model
    device = "cpu"
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)
    
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
    
    # Height adjustment
    if args.height_adjust:
        print("Adjusting height...")
        ground_offset = 0.0
        lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
        # Recompute body_pos and body_rot after height adjustment
        body_pos, body_rot = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
    
    # Root origin offset
    if args.root_origin_offset:
        print("Applying root origin offset...")
        root_pos[:, :2] -= root_pos[0, :2]
        # Recompute body_pos and body_rot after root offset
        body_pos, body_rot = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
            torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
        )
    
    # Prepare motion data - Version 1: matching smplx_to_robot.py structure
    motion_data_v1 = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    
    # Prepare motion data - Version 2: with body_pos and other necessary elements
    motion_data_v2 = {
        "fps": aligned_fps,
        "body_pos": body_pos.detach().cpu().numpy(),  # Global body positions (TxNx3)
        "body_rot": body_rot.detach().cpu().numpy(),  # Global body rotations (TxNx4, quaternions)
        "link_body_list": body_names,
    }

    retargeter.calculate_error_statistics_and_plot(save_path='/Users/huangxiansheng/Desktop/retarget/error_logs/Stefanos_1os_antrikos_karsilamas_C3D_stageii_v3_error_stats.json')
    
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
        
        print(f"\nSaved both versions of motion data:")
        print(f"  Version 1 (simple, matching smplx_to_robot.py):")
        print(f"    - Saved to: {save_path_v1}")
        print(f"    - {num_frames} frames at {aligned_fps} fps")
        print(f"    - root_pos shape: {root_pos.shape}")
        print(f"    - root_rot shape: {root_rot.shape}")
        print(f"    - dof_pos shape: {dof_pos.shape}")
        print(f"    - local_body_pos: None")
        print(f"    - link_body_list: None")
        print(f"  Version 2 (with body_pos):")
        print(f"    - Saved to: {save_path_v2}")
        print(f"    - {num_frames} frames at {aligned_fps} fps")
        print(f"    - body_pos shape: {body_pos.shape}")
        print(f"    - body_rot shape: {body_rot.shape}")
        print(f"    - {len(body_names)} body links")
        
        print("\nDone!")
