import argparse
import pathlib
import os
import multiprocessing as mp

import numpy as np
from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
import gc
import time
import psutil
import tracemalloc


def check_memory(threshold_gb=3):  # adjust based on your available memory
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent


def process_file(smplx_file_path, tgt_file_path, tgt_robot, SMPLX_FOLDER, tgt_folder, total_files, 
                 height_adjust=False, root_origin_offset=False, contact_filter=False, verbose=False):
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
        
    # Initial checks (with optional logging)
    log_memory("Initial memory usage")
    
    num_pause = 0
    while check_memory():
        print(f"[PAUSE] Paused processing {smplx_file_path} to prevent memory overflow. num_pause: {num_pause}")
        time.sleep(60*2)
        num_pause += 1
        if num_pause > 10:
            print(f"[ERROR] Memory usage is still high after 10 pauses. Exiting.")
            return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
        mocap_frame_rate = smplx_data["mocap_frame_rate"]
        log_memory("After loading SMPL-X data")
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return
    
  
    tgt_fps = 30
    try:
        smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    
    # Initialize retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=tgt_robot,
        aligned_fps=aligned_fps,
        contact_filter=contact_filter,
    )
    
    # Retarget all frames
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

    log_memory("After retargeting")
    
    device = "cpu"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)

    try:
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        num_frames = root_pos.shape[0]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return

    # Compute local body positions (with zero root pos/rot)
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    log_memory("After forward kinematics")

    body_names = kinematics_model.body_names
    
    # Compute global body positions and rotations (with actual root pos/rot)
    body_pos, body_rot = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )  # Shape: TxNx3 for body_pos, TxNx4 for body_rot
    
    # height adjust to ensure the lowest part is on the ground
    ground_offset = 0.0
    if height_adjust:
        lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
    
    # offset using the first frame
    if root_origin_offset:
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
    
    # height adjust for tracking links
    if height_adjust:
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

    # offset using the first frame for tracking links
    if root_origin_offset:
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
    num_frames_tracking = len(tracking_links_pos_list)
    
    # Create reverse mapping for lookup
    OPTITRACK_TO_ROBOT = {v: k for k, v in ROBOT_TO_OPTITRACK.items()}
    
    # Create arrays for pos and quat: (num_frames, num_links, 3/4)
    pos_array = np.zeros((num_frames_tracking, num_links, 3), dtype=np.float32)
    quat_array = np.zeros((num_frames_tracking, num_links, 4), dtype=np.float32)
    frame_id_array = np.arange(num_frames_tracking, dtype=np.int32)
    
    # Fill in the data from tracking_links_pos_list
    # link_names contains OptiTrack names, but we need robot link names to access the data
    for frame_idx in range(num_frames_tracking):
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

    # Prepare file paths for both versions
    base_path = tgt_file_path
    if base_path.endswith('.pkl'):
        base_path = base_path[:-4]  # Remove .pkl extension
    
    save_path_v1 = base_path + '_v1.pkl'
    save_path_v2 = base_path + '_v2.pkl'
    
    os.makedirs(os.path.dirname(save_path_v1), exist_ok=True)
    
    # Save version 1
    with open(save_path_v1, "wb") as f:
        pickle.dump(motion_data_v1, f)
    
    # Save version 2
    with open(save_path_v2, "wb") as f:
        pickle.dump(motion_data_v2, f)
        
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('_v1.pkl')])  # Count v1 files
    print(f"Processed {done}/{total_files}: {save_path_v1}")
    
    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()
        
    # clean cache
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
    )
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
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
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=4, type=int)
    args = parser.parse_args()
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions_paths = [hard_motions_folder / "0.txt", 
                          hard_motions_folder / "1.txt"]
    hard_motions = []
    for hard_motions_path in hard_motions_paths:
        if hard_motions_path.exists():
            with open(hard_motions_path, "r") as f:
                for line in f:
                    if "Motion:" in line:
                        motion_path = line.split(":")[1].strip()
                    else:
                        continue
                    motion_path = motion_path.split(",")[0].strip().split(".")[0]
                    hard_motions.append(motion_path)
                
                
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")
                # Check if both v1 and v2 exist
                base_path = tgt_file_path[:-4] if tgt_file_path.endswith('.pkl') else tgt_file_path
                save_path_v1 = base_path + '_v1.pkl'
                save_path_v2 = base_path + '_v2.pkl'
                if (not os.path.exists(save_path_v1) or not os.path.exists(save_path_v2) or args.override):
                    args_list.append((smplx_file_path, tgt_file_path, args.robot, SMPLX_FOLDER, tgt_folder,
                                    args.height_adjust, args.root_origin_offset, args.contact_filter))
    print("full args_list:", len(args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    hard_motions_lower = {m.lower() for m in hard_motions}
    exclude_lower = [c.lower() for c in exclude_file_content]
    new_args_list = []
    for arguments in args_list:
        motion_name = arguments[0].split("/")[-1].split('.')[0]
        motion_lower = motion_name.lower()
        if motion_lower in hard_motions_lower:
            continue
        if any(content in motion_lower for content in exclude_file_content):
            continue
        new_args_list.append(arguments)
    args_list = new_args_list
    
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    with mp.Pool(args.num_cpus) as pool:
        pool.starmap(process_file, [args + (total_files, verbose) for args in args_list])

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
