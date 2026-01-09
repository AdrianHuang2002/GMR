import argparse
import pathlib
import os
import time
import platform

import torch
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    load_smplx_file,
    get_smplx_data_offline_fast,
)
from gs_env.sim.envs.config.registry import EnvArgsRegistry
import gs_env.sim.envs as envs


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
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )


    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )

    # Align fps
    tgt_fps = 30
    show_viewer = True
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
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def step_env_with_qpos(qpos, tracking_links_pos):
        """
        qpos: [root_pos(3), root_quat(4 wxyz), dof_pos(...)]
        tracking_links_pos: dict[link_name] = (pos(3,), quat(4,))
        """
        qpos_t = torch.tensor(qpos, device=env.device, dtype=torch.float32)

        root_pos = qpos_t[0:3]
        root_quat = qpos_t[3:7]
        dof_pos = qpos_t[7:]

        env.robot.set_state(
            pos=root_pos,
            quat=root_quat,
            dof_pos=dof_pos,
        )

        if tracking_links_pos is not None:
            for link_name, (pos, quat) in tracking_links_pos.items():
                pos_t = torch.tensor(pos, device=env.device, dtype=torch.float32)[None, :]
                quat_t = torch.tensor(quat, device=env.device, dtype=torch.float32)[None, :]
                env.scene.set_obj_pose(link_name, pos=pos_t, quat=quat_t)  # type: ignore

        env.scene.scene.step(refresh_visualizer=False)  # type: ignore

    def motion_loop():
        fps_counter = 0
        fps_start_time = time.time()
        fps_display_interval = 2.0 

        i = 0  

        while True:
            # Advance frame index
            if args.loop:
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
            retarget.update_targets(smplx_frame)
            qpos = retarget.retarget()
            # get tracking links positions
            tracking_links_pos = retarget.get_human_tracking_targets()

            # Apply to env + markers
            step_env_with_qpos(qpos, tracking_links_pos)

            # Save qpos if requested
            if args.save_path is not None:
                qpos_list.append(qpos.copy())

    if platform.system() == "Darwin" and show_viewer:
        import threading

        threading.Thread(target=motion_loop, daemon=True).start()
        env.scene.scene.viewer.run()  # type: ignore
    else:
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
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
