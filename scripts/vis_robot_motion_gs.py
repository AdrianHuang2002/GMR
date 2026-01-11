import argparse
import os
import pathlib
import time
import pickle

import numpy as np
import torch

from general_motion_retargeting import load_robot_motion
from gs_env.sim.envs.config.registry import EnvArgsRegistry
import gs_env.sim.envs as gs_envs


def play_robot_motion(
    env: gs_envs.MotionEnv,
    motion_root_pos,
    motion_root_rot,
    motion_dof_pos,
    motion_fps: float,
    show_viewer: bool = True,
    tracking_link_names_v2=None,
    tracking_link_pos_v2=None,
    tracking_link_quat_v2=None,
) -> None:
    """Play a saved robot motion in a Genesis MotionEnv.
    
    Args:
        tracking_link_names_v2: List/tuple of tracking link names from v2 data (OptiTrack names)
        tracking_link_pos_v2: Tensor of shape (num_frames, num_links, 3) from v2 data
        tracking_link_quat_v2: Tensor of shape (num_frames, num_links, 4) from v2 data
    """
    # Map OptiTrack names back to robot link names (reverse mapping of ROBOT_TO_OPTITRACK)
    # This matches the mapping in smplx_to_robot_no_viewer.py
    OPTITRACK_TO_ROBOT = {
        "LeftFoot": "left_ankle_roll_link",
        "RightFoot": "right_ankle_roll_link",
        "LeftHand": "left_wrist_yaw_link",
        "RightHand": "right_wrist_yaw_link",
        "Spine1": "torso_link",
        "Hips": "pelvis",
    }
    
    # Map tracking_link_names_v2 from OptiTrack names to robot link names
    # The v2 data stores link names as OptiTrack names, but env.scene.set_obj_pose needs robot link names
    if tracking_link_names_v2 is not None:
        tracking_link_names_v2 = [OPTITRACK_TO_ROBOT.get(name, name) for name in tracking_link_names_v2]

    def run() -> None:
        frame_idx = 0
        last_update_time = time.time()
        while True:
            pos_t = torch.tensor(motion_root_pos[frame_idx], dtype=torch.float32, device=env.device)
            quat_t = torch.tensor(motion_root_rot[frame_idx], dtype=torch.float32, device=env.device)
            dof_t = torch.tensor(motion_dof_pos[frame_idx], dtype=torch.float32, device=env.device)

            # set state and update
            env.robot.set_state(pos=pos_t, quat=quat_t, dof_pos=dof_t)
            env.update_buffers()

            # Draw tracking links from v2 data using env.scene.set_obj_pose
            if (tracking_link_names_v2 is not None and 
                tracking_link_pos_v2 is not None and 
                tracking_link_quat_v2 is not None and
                frame_idx < len(tracking_link_pos_v2)):
                
                for link_idx, robot_link_name in enumerate(tracking_link_names_v2):
                    if link_idx < tracking_link_pos_v2.shape[1]:
                        pos = tracking_link_pos_v2[frame_idx, link_idx].cpu().numpy()
                        quat = tracking_link_quat_v2[frame_idx, link_idx].cpu().numpy()
                        pos_t_link = torch.tensor(pos, device=env.device, dtype=torch.float32)[None, :]
                        quat_t_link = torch.tensor(quat, device=env.device, dtype=torch.float32)[None, :]
                        env.scene.set_obj_pose(robot_link_name, pos=pos_t_link, quat=quat_t_link)  # type: ignore

            if show_viewer:
                env.scene.scene.step(refresh_visualizer=False)

            # maintain playback rate
            while time.time() - last_update_time < 1.0 / motion_fps:
                time.sleep(0.001)
            last_update_time = time.time()

            # loop frames
            frame_idx = (frame_idx + 1) % len(motion_root_pos)

    try:
        if show_viewer:
            import threading
            # run sim loop in background, viewer on main thread (Mac-friendly)
            threading.Thread(target=run, daemon=True).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            run()
    except KeyboardInterrupt:
        print("\n[GenesisViewer] Stopped by user.")


def load_motion_file(robot_motion_path, v1_path=None, v2_path=None):
    """
    Load motion file, supporting both v1 and v2 formats.
    Returns both v1 data (for robot visualization) and v2 data (for tracking links).
    
    v1 format: fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list, foot_contact
    v2 format: fps, link_names, pos, quat, frame_id, foot_contact
    """
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    # Determine version from filename or content
    motion_path = pathlib.Path(robot_motion_path)
    is_v2 = motion_path.name.endswith("_v2.pkl")
    is_v1 = motion_path.name.endswith("_v1.pkl")
    
    # Load the file to check format
    with open(robot_motion_path, "rb") as f:
        motion_data = pickle.load(f)
    
    # Detect version from content if not clear from filename
    if not is_v1 and not is_v2:
        if "link_names" in motion_data and "pos" in motion_data and "quat" in motion_data:
            is_v2 = True
        elif "root_pos" in motion_data and "root_rot" in motion_data and "dof_pos" in motion_data:
            is_v1 = True
    
    motion_data_v1 = None
    motion_data_v2 = None
    
    if is_v1:
        motion_data_v1 = motion_data
        # Try to find corresponding v2 file
        if v2_path is None:
            base_path = str(motion_path).replace("_v1.pkl", "_v2.pkl").replace("_v1", "_v2")
            if os.path.exists(base_path):
                v2_path = base_path
        
        if v2_path and os.path.exists(v2_path):
            print(f"Loading V2 file for tracking links: {v2_path}")
            with open(v2_path, "rb") as f:
                motion_data_v2 = pickle.load(f)
    
    elif is_v2:
        motion_data_v2 = motion_data
        # Try to find corresponding v1 file
        base_path = None
        if v1_path is None:
            # Auto-detect v1 file path
            base_path = str(motion_path).replace("_v2.pkl", "_v1.pkl").replace("_v2", "_v1")
            if os.path.exists(base_path):
                v1_path = base_path
            else:
                # Try without replacing, just remove _v2
                alt_path = str(motion_path).replace("_v2.pkl", ".pkl")
                if os.path.exists(alt_path):
                    # Check if it's v1 format
                    with open(alt_path, "rb") as f_check:
                        check_data = pickle.load(f_check)
                    if "root_pos" in check_data:
                        v1_path = alt_path
        
        if v1_path is None or not os.path.exists(v1_path):
            error_msg = (
                f"V2 format file provided but corresponding V1 file not found. "
                f"V1 file is required for robot visualization. "
                f"Please provide a V1 file path using --v1_path"
            )
            if base_path:
                error_msg += f" or ensure {base_path} exists."
            raise FileNotFoundError(error_msg)
        
        print(f"Loading V1 file for robot visualization: {v1_path}")
        with open(v1_path, "rb") as f:
            motion_data_v1 = pickle.load(f)
    
    # Load v1 format data (required for robot visualization)
    if motion_data_v1 is None or "root_pos" not in motion_data_v1:
        raise ValueError(f"V1 format file is required but not found or invalid. V1 must contain root_pos/root_rot/dof_pos.")
    
    motion_fps = motion_data_v1["fps"]
    motion_root_pos = motion_data_v1["root_pos"]
    # Convert from xyzw to wxyz format
    if motion_data_v1["root_rot"].shape[1] == 4:
        motion_root_rot = motion_data_v1["root_rot"][:, [3, 0, 1, 2]]
    else:
        motion_root_rot = motion_data_v1["root_rot"]
    motion_dof_pos = motion_data_v1["dof_pos"]
    motion_local_body_pos = motion_data_v1.get("local_body_pos", None)
    motion_link_body_list = motion_data_v1.get("link_body_list", None)
    
    # Load v2 format data (optional, for tracking links visualization)
    tracking_link_names_v2 = None
    tracking_link_pos_v2 = None
    tracking_link_quat_v2 = None
    
    if motion_data_v2 is not None:
        tracking_link_names_v2 = motion_data_v2.get("link_names", None)
        tracking_link_pos_v2 = motion_data_v2.get("pos", None)  # (num_frames, num_links, 3)
        tracking_link_quat_v2 = motion_data_v2.get("quat", None)  # (num_frames, num_links, 4)
        
        if tracking_link_names_v2 is not None:
            tracking_link_names_v2 = list(tracking_link_names_v2)  # Convert tuple to list
            print(f"V2 file contains {len(tracking_link_names_v2)} tracking links: {tracking_link_names_v2}")
    
    return (
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
        tracking_link_names_v2,
        tracking_link_pos_v2,
        tracking_link_quat_v2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True,
                       help="Path to motion file (supports both _v1.pkl and _v2.pkl formats)")
    parser.add_argument("--v1_path", type=str, default=None,
                       help="Optional: Path to V1 file if V2 file is provided (auto-detected if not specified)")
    parser.add_argument("--v2_path", type=str, default=None,
                       help="Optional: Path to V2 file if V1 file is provided (auto-detected if not specified)")
    args = parser.parse_args()

    show_viewer = True
    robot_motion_path = args.robot_motion_path
    
    # load motion file (supports both v1 and v2)
    (
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
        tracking_link_names_v2,
        tracking_link_pos_v2,
        tracking_link_quat_v2,
    ) = load_motion_file(robot_motion_path, args.v1_path, args.v2_path)

    # build Genesis env
    env_args = EnvArgsRegistry["g1_motion"]
    device = torch.device("cpu")
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=device,
        eval_mode=True,
    )
    env.reset()

    play_robot_motion(
        env=env,
        motion_root_pos=motion_root_pos,
        motion_root_rot=motion_root_rot,
        motion_dof_pos=motion_dof_pos,
        motion_fps=motion_fps,
        show_viewer=show_viewer,
        tracking_link_names_v2=tracking_link_names_v2,
        tracking_link_pos_v2=tracking_link_pos_v2,
        tracking_link_quat_v2=tracking_link_quat_v2,
    )
