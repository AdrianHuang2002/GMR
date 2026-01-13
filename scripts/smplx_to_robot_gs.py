import argparse
import pathlib
import os
import time

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)
from gs_env.sim.envs.config.registry import EnvArgsRegistry
import gs_env.sim.envs as envs


HUMAN_TO_ROBOT_TRACKING_DICT = {
    "pelvis": "pelvis",
    "spine3": "torso_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
        ],
        default="unitree_g1",
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )

    parser.add_argument(
        "--view",
        default=False,
        action="store_true",
        help="View the motion.",
    )


    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # Align fps
    tgt_fps = 30
    show_viewer = args.view
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    env_args = EnvArgsRegistry["g1_motion"]
    tracking_link_names = getattr(env_args, "tracking_link_names", [])

    envclass = getattr(envs, env_args.env_name)
    env = envclass(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device(device),
        eval_mode=True,
    )
    env.reset()

    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        aligned_fps = aligned_fps,
    )

    qpos_list = []
    tracking_links_pos_list = []
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def motion_loop():
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0 

        i = 0  

        while True:
            # Advance frame index
            if args.view:
                i = (i + 1) % len(smplx_data_frames)
            else:
                i += 1
                if i >= len(smplx_data_frames):
                    break

            # FPS measurements
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            # Current SMPLX frame
            smplx_frame = smplx_data_frames[i]

            # Retarget
            scaled_human_data = retarget.process_human_data(smplx_frame)
            qpos = retarget.retarget(scaled_human_data)
            qpos_t = torch.tensor(qpos, device=env.device, dtype=torch.float32)
            tracking_links_pos = {}

            for human_name, robot_name in HUMAN_TO_ROBOT_TRACKING_DICT.items():
                if human_name in scaled_human_data.keys():
                    pos, quat = scaled_human_data[human_name]
                    pos_t = torch.tensor(pos, device=env.device, dtype=torch.float32)
                    quat_t = torch.tensor(quat, device=env.device, dtype=torch.float32)
                    if "ankle" in robot_name:
                        offset = torch.tensor([-0.1, 0, 0.02], device=env.device, dtype=torch.float32)
                        pos_t += R.from_quat(quat_t, scalar_first=True).apply(offset)
                    if "torso" in robot_name:
                        offset = np.array([-0.0039635, 0.0, 0.044], dtype=float)
                        pos_t = torch.tensor(scaled_human_data["pelvis"][0]) + R.from_quat(quat_t, scalar_first=True).apply(offset)
                    tracking_links_pos[robot_name] = (pos_t, quat_t)

            if args.view:
                env.robot.set_state(
                    pos=qpos_t[:3],
                    quat=qpos_t[3:7],
                    dof_pos=qpos_t[7:],
                )

                if tracking_links_pos is not None:
                    for link_name, (pos, quat) in tracking_links_pos.items():
                        env.scene.set_obj_pose(link_name, pos=pos[None, :], quat=quat[None, :])  # type: ignore

                env.scene.scene.step()

            # Save qpos if requested
            if args.save_path is not None:
                qpos_list.append(qpos.copy())
                tracking_links_pos_list.append(tracking_links_pos.copy())

    motion_loop()

    # Save motion to pickle if requested
    if args.save_path is not None and len(qpos_list) > 0:
        import pickle

        root_pos = np.array([q[:3] for q in qpos_list])
        # convert from wxyz -> xyzw
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])
        dof_pos = np.array([q[7:] for q in qpos_list])
        local_body_pos = None
        body_names = None

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "tracking_links_pos": tracking_links_pos_list,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
