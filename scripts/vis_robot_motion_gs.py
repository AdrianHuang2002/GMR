import argparse
import os
import time

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
    tracking_link_indices=None,
) -> None:
    """Play a saved robot motion in a Genesis MotionEnv."""

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

            if tracking_link_indices is not None:
                link_pos_all = env.robot.link_positions[0]
                link_quat_all = env.robot.link_quaternions[0]
                print(f"\n[Frame {frame_idx}]")
                for name, idx in zip(tracking_link_names, tracking_link_indices):
                    pos = link_pos_all[idx]
                    quat = link_quat_all[idx]
                    print(f"  {name:25s} pos={pos.tolist()}   quat={quat.tolist()}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    args = parser.parse_args()

    show_viewer = True
    robot_motion_path = args.robot_motion_path
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    # load GMR-style motion file
    (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    ) = load_robot_motion(robot_motion_path)

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
    tracking_link_names = getattr(env_args, "tracking_link_names", [])
    link_name_to_idx = {
        link.name: i for i, link in enumerate(env.robot.robot.links)
    }
    tracking_link_indices = [link_name_to_idx[name] for name in tracking_link_names]

    play_robot_motion(
        env=env,
        motion_root_pos=motion_root_pos,
        motion_root_rot=motion_root_rot,
        motion_dof_pos=motion_dof_pos,
        motion_fps=motion_fps,
        show_viewer=show_viewer,
        tracking_link_indices=tracking_link_indices,
    )
