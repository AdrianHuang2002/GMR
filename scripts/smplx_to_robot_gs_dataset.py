import argparse
import pathlib
import os
import multiprocessing as mp
import pickle

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from natsort import natsorted

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)
from general_motion_retargeting.kinematics_model import KinematicsModel

HUMAN_TO_ROBOT_TRACKING_DICT = {
    "pelvis": "pelvis",
    "spine3": "torso_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}


HERE = pathlib.Path(__file__).parent


def process_file(
    smplx_file_path,
    tgt_file_path_after,
    tgt_file_path_before,
    tgt_robot,
    SMPLX_FOLDER,
    tgt_folder_after,
    total_files,
    height_adjust=False,
    root_origin_offset=False,
    contact_filter=False,
    verbose=False,
):
    """Process a single file for dataset mode (no viewer)."""
 
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file_path, SMPLX_FOLDER
    )
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
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
        # Use update_targets to store scaled data, then retarget
        scaled_human_data = retarget.process_human_data(smplx_frame_data)
        qpos = retarget.retarget(scaled_human_data)
        foot_contact_list.append(
            np.asarray(retarget.last_foot_contact, dtype=np.float32).copy()
        )
        qpos_list.append(qpos)

        # Build tracking_links_pos manually from scaled_human_data
        tracking_links_pos = {}
        for human_name, robot_name in HUMAN_TO_ROBOT_TRACKING_DICT.items():
            if human_name in scaled_human_data.keys():
                pos, quat = scaled_human_data[human_name]
                pos = np.asarray(pos, dtype=np.float32)
                quat = np.asarray(quat, dtype=np.float32)
                # Apply same offsets as in single file version
                if "ankle" in robot_name:
                    offset = np.array([-0.1, 0, 0.02], dtype=np.float32)
                    pos = pos + R.from_quat(quat, scalar_first=True).apply(offset)
                if "torso" in robot_name:
                    offset = np.array([-0.0039635, 0.0, 0.044], dtype=np.float32)
                    pos = np.asarray(scaled_human_data["pelvis"][0], dtype=np.float32) + R.from_quat(
                        quat, scalar_first=True
                    ).apply(offset)
                tracking_links_pos[robot_name] = (pos, quat)
        tracking_links_pos_list.append(tracking_links_pos)

    qpos_list = np.array(qpos_list)
    foot_contact_array = np.array(foot_contact_list)  # Shape: (num_frames, 2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    num_frames = root_pos.shape[0]

    # Compute local body positions (with zero root pos/rot)
    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    body_names = kinematics_model.body_names

    # Compute global body positions and rotations (with actual root pos/rot)
    body_pos, body_rot = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
        torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
    )  # Shape: TxNx3 for body_pos, TxNx4 for body_rot

    # height adjust to ensure the lowest part is on the ground
    ground_offset = 0.0
    if height_adjust:
        lowest_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset

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

    # Use the 6 tracking links from HUMAN_TO_ROBOT_TRACKING_DICT
    link_names = list(HUMAN_TO_ROBOT_TRACKING_DICT.values())
    num_links = len(link_names)
    num_frames_tracking = len(tracking_links_pos_list)

    # Create arrays for pos and quat: (num_frames, num_links, 3/4)
    pos_array = np.zeros((num_frames_tracking, num_links, 3), dtype=np.float32)
    quat_array = np.zeros((num_frames_tracking, num_links, 4), dtype=np.float32)

    # Fill in the data from tracking_links_pos_list
    for frame_idx in range(num_frames_tracking):
        frame_data = tracking_links_pos_list[frame_idx]
        for link_idx, robot_link_name in enumerate(link_names):
            if robot_link_name in frame_data:
                pos, quat = frame_data[robot_link_name]
                pos_array[frame_idx, link_idx] = np.asarray(pos, dtype=np.float32)
                quat_array[frame_idx, link_idx] = np.asarray(quat, dtype=np.float32)

    # Convert numpy arrays to torch Tensors
    pos_tensor = torch.from_numpy(pos_array).to(dtype=torch.float32)
    quat_tensor = torch.from_numpy(quat_array).to(dtype=torch.float32)
    foot_contact_tensor = torch.from_numpy(foot_contact_array).to(dtype=torch.float32)

    motion_data_v2 = {
        "fps": aligned_fps,
        "link_names": tuple(link_names),  # Robot link names from HUMAN_TO_ROBOT_TRACKING_DICT
        "pos": pos_tensor,  # Shape: (num_frames, num_links, 3)
        "quat": quat_tensor,  # Shape: (num_frames, num_links, 4)
        "foot_contact": foot_contact_tensor,  # Shape: (num_frames, 2) - foot contact for left and right feet
    }

    # Prepare file paths for both versions in separate target directories
    dir_after = os.path.dirname(tgt_file_path_after)
    dir_before = os.path.dirname(tgt_file_path_before)
    if dir_after:
        os.makedirs(dir_after, exist_ok=True)
    if dir_before:
        os.makedirs(dir_before, exist_ok=True)

    # Save version 1 (after retarget) - robot motion format
    with open(tgt_file_path_after, "wb") as f:
        pickle.dump(motion_data_v1, f)

    # Save version 2 (before retarget) - tracking links format
    with open(tgt_file_path_before, "wb") as f:
        pickle.dump(motion_data_v2, f)

    # Progress print based on tgt_folder_after (count files in after directory)
    done = 0
    if os.path.exists(tgt_folder_after):
        for root, _, files in os.walk(tgt_folder_after):
            done += len([f for f in files if f.endswith(".pkl")])
    print(f"Processed {done}/{total_files}: {tgt_file_path_after}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--src_folder",
        type=str,
        required=True,
        help="Source folder containing SMPLX motion files (.pkl or .npz).",
    )
    parser.add_argument(
        "--tgt_folder_after",
        type=str,
        required=True,
        help="Target folder for v1 format files (after retarget - robot motion with root_pos, root_rot, dof_pos).",
    )
    parser.add_argument(
        "--tgt_folder_before",
        type=str,
        required=True,
        help="Target folder for v2 format files (before retarget - tracking links with robot link names).",
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
        action="store_true",
        help="Filter contacts to ensure the robot feet on ground.",
    )

    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Override existing files if they already exist.",
    )

    parser.add_argument(
        "--num_cpus",
        default=4,
        type=int,
        help="Number of CPU cores to use for parallel processing.",
    )

    args = parser.parse_args()

    # Print the total number of CPUs
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")

    src_folder = args.src_folder
    tgt_folder_after = args.tgt_folder_after
    tgt_folder_before = args.tgt_folder_before

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    hard_motions_paths = [hard_motions_folder / "0.txt", hard_motions_folder / "1.txt"]
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
                # Create target paths in both directories, preserving relative structure
                rel_path = os.path.relpath(smplx_file_path, src_folder)
                tgt_file_path_after = os.path.join(tgt_folder_after, rel_path).replace(".npz", ".pkl")
                tgt_file_path_before = os.path.join(tgt_folder_before, rel_path).replace(".npz", ".pkl")

                # Check if both files exist
                if (
                    not os.path.exists(tgt_file_path_after)
                    or not os.path.exists(tgt_file_path_before)
                    or args.override
                ):
                    args_list.append(
                        (
                            smplx_file_path,
                            tgt_file_path_after,
                            tgt_file_path_before,
                            args.robot,
                            SMPLX_FOLDER,
                            tgt_folder_after,
                            args.height_adjust,
                            args.root_origin_offset,
                            args.contact_filter,
                        )
                    )

    print(f"full args_list: {len(args_list)}")

    # remove hard and infeasible motions
    hard_motions = {m for m in hard_motions}
    exclude_dirs = {"BMLrub", "EKUT"}
    exclude_keywords = ["crawl", "_lie", "upstairs", "downstairs"]
    new_args_list = []
    src_root = pathlib.Path(src_folder)

    for arguments in args_list:
        smplx_file_path = arguments[0]
        motion_name = pathlib.Path(smplx_file_path).stem
        # parts relative to src_folder, e.g. ("CMU", "133", "133_14_stageii.npz")
        rel_parts = pathlib.Path(smplx_file_path).relative_to(src_root).parts
        top_dir = rel_parts[0] if rel_parts else ""
        rel_parts_str = "_".join(rel_parts).rsplit(".", 1)[0]
        # 1) Filter hard motions (exact filename match, no path, no ext)
        if rel_parts_str in hard_motions:
            continue
        # 2) Filter exclude directories (dir name match)
        if top_dir in exclude_dirs:
            continue
        # 3) Filter other exclude keywords in filename
        if any(k in motion_name.lower() for k in exclude_keywords):
            continue
        new_args_list.append(arguments)

    args_list = new_args_list

    print(f"new args_list: {len(args_list)}")

    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")

    if total_files > 0:
        with mp.Pool(args.num_cpus) as pool:
            pool.starmap(process_file, [args + (total_files, False) for args in args_list])

        print(f"Done. Saved v1 (after retarget) to: {tgt_folder_after}")
        print(f"Done. Saved v2 (before retarget) to: {tgt_folder_before}")
    else:
        print("No files to process.")


if __name__ == "__main__":
    main()
