from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_folder", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",   # can be file or directory
    )

    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_folder = args.robot_motion_folder

    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(
            f"Motion data dir {robot_motion_folder} does not exist."
        )

    motion_files = [f for f in os.listdir(robot_motion_folder) if f.endswith(".pkl")]
    motion_files = sorted(motion_files)
    print(f"Found {len(motion_files)} motion files in {robot_motion_folder}, converting...")

    base_video_path = args.video_path
    is_dir_output = os.path.isdir(base_video_path)

    # If video_path is not an existing directory but has a slash, treat its directory part as target dir
    if not is_dir_output:
        out_dir = os.path.dirname(base_video_path)
        if out_dir == "":
            out_dir = "videos"
        os.makedirs(out_dir, exist_ok=True)

    for motion_file in tqdm(motion_files):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        (
            motion_data,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            motion_local_body_pos,
            motion_link_body_list,
        ) = load_robot_motion(motion_path)

        # Decide output video path for THIS motion file
        if is_dir_output:
            os.makedirs(base_video_path, exist_ok=True)
            video_path = os.path.join(
                base_video_path, motion_file.replace(".pkl", ".mp4")
            )
        else:
            # base_video_path is a file path; keep its directory, change filename per motion
            out_dir = os.path.dirname(base_video_path)
            base_name = motion_file.replace(".pkl", ".mp4")
            video_path = os.path.join(out_dir, base_name)

        print(f"\nConverting {motion_file} → {video_path}")

        env = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            camera_follow=False,
            record_video=args.record_video,
            video_path=video_path,
            keyboard_callback=None,
        )

        # Play this motion ONCE
        num_frames = len(motion_root_pos)
        for frame_idx in range(num_frames):
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )

        env.close()

    print("All motions converted.")
