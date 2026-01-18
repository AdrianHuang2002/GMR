import argparse
import pathlib
import os
import time
import pickle
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import smplx

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

def retarget_smplx(smplx_data, fps, actual_human_height, env, view):

    # Initialize the retargeting system
    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        aligned_fps = fps,
    )

    raw_tracking_link_names = [robot_name for _, robot_name in HUMAN_TO_ROBOT_TRACKING_DICT.items()]
    raw_tracking_link_pos_global = []
    raw_tracking_link_quat_global = []
    raw_pos_list = []
    raw_quat_list = []
    foot_links_idx = (
        raw_tracking_link_names.index("left_ankle_roll_link"),
        raw_tracking_link_names.index("right_ankle_roll_link"),
    )
    raw_motion_data = {
        "fps": fps,
        "link_names": raw_tracking_link_names,
        "dof_names": env.dof_names,
        "foot_link_indices": foot_links_idx,
    }
    base_idx = raw_tracking_link_names.index("pelvis")
    retargeted_tracking_link_names = [link.name for link in env.robot.robot.links]
    retargeted_tracking_link_pos_global = []
    retargeted_tracking_link_quat_global = []
    tracking_link_pos = torch.zeros_like(env.tracking_link_pos_global)[0]
    tracking_link_quat = torch.zeros_like(env.tracking_link_quat_local_yaw)[0]
    pos_list = []
    quat_list = []
    dof_pos_list = []
    foot_contact_list = []
    foot_last_pos = None
    foot_contact = torch.ones(2, dtype=torch.float32)
    retargeted_motion_data = {
        "fps": fps,
        "link_names": retargeted_tracking_link_names,
        "dof_names": env.dof_names,
        "foot_link_indices": env.robot.foot_links_idx,
    }

    frame_idx = 0 
    frame_counter = 0
    retarget_start_time = time.time()
    speed_measurement_interval = 2.0 

    while True:
        # Advance frame index
        if view:
            frame_idx = (frame_idx + 1) % len(smplx_data)
            # FPS measurements
            frame_counter += 1
            current_time = time.time()
            if current_time - retarget_start_time >= speed_measurement_interval:
                actual_fps = frame_counter / (current_time - retarget_start_time)
                print(f"Actual retargeting FPS: {actual_fps:.2f}")
                frame_counter = 0
                retarget_start_time = current_time

        else:
            frame_idx += 1
            if frame_idx >= len(smplx_data):
                break

        # Current SMPLX frame
        smplx_frame = smplx_data[frame_idx]

        # Retarget
        scaled_human_data = retargeter.process_human_data(smplx_frame)
        qpos = retargeter.retarget(scaled_human_data)
        qpos_t = torch.tensor(qpos, device=env.device, dtype=torch.float32)

        for j, (human_name, robot_name) in enumerate(HUMAN_TO_ROBOT_TRACKING_DICT.items()):
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
                tracking_link_pos[j] = pos_t
                tracking_link_quat[j] = quat_t

        foot_pos = tracking_link_pos[foot_links_idx, :]
        if foot_last_pos is not None:
            foot_vel = torch.clamp(
                (torch.norm((foot_pos[..., :2] - foot_last_pos[..., :2]) * fps, dim=-1) - 0.2)
                / 0.2,
                0.0,
                1.0,
            )
            foot_lift = torch.clamp((foot_pos[:, 2] - 0.2) / 0.2, 0.0, 1.0)
            foot_not_contact = (foot_lift + foot_vel).clamp(0.0, 1.0)
            foot_contact = 1 - foot_not_contact
        foot_last_pos = foot_pos.clone()
        foot_contact_list.append(foot_contact.clone())

        raw_tracking_link_pos_global.append(tracking_link_pos.clone())
        raw_tracking_link_quat_global.append(tracking_link_quat.clone())
        raw_pos_list.append(tracking_link_pos[base_idx].clone())
        raw_quat_list.append(tracking_link_quat[base_idx].clone())

        env.robot.set_state(
            pos=qpos_t[:3],
            quat=qpos_t[3:7],
            dof_pos=qpos_t[7:],
        )
        env.update_buffers()

        pos_list.append(qpos_t[:3].clone())
        quat_list.append(qpos_t[3:7].clone())
        dof_pos_list.append(qpos_t[7:].clone())
        retargeted_tracking_link_pos_global.append(env.link_positions[0].clone())
        retargeted_tracking_link_quat_global.append(env.link_quaternions[0].clone())

        if view:
            env.scene.scene.clear_debug_objects()
            for j, link_name in enumerate(raw_tracking_link_names):
                pos = tracking_link_pos[j]
                quat = tracking_link_quat[j]
                env.scene.set_obj_pose(link_name, pos=pos[None, :], quat=quat[None, :])  # type: ignore
            for i in range(len(foot_links_idx)):
                env.scene.scene.draw_debug_arrow(
                    foot_pos[i],
                    foot_contact[i] * torch.tensor([0.0, 0.0, 0.5]),
                    radius=0.01,
                    color=(0.0, 0.0, 1.0),
                )
            env.scene.scene.step()

    raw_motion_data["pos"] = torch.stack(raw_pos_list).numpy()
    raw_motion_data["quat"] = torch.stack(raw_quat_list).numpy()
    raw_motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
    raw_motion_data["link_pos"] = torch.stack(raw_tracking_link_pos_global).numpy()
    raw_motion_data["link_quat"] = torch.stack(raw_tracking_link_quat_global).numpy()
    raw_motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

    retargeted_motion_data["pos"] = torch.stack(pos_list).numpy()
    retargeted_motion_data["quat"] = torch.stack(quat_list).numpy()
    retargeted_motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
    retargeted_motion_data["link_pos"] = torch.stack(retargeted_tracking_link_pos_global).numpy()
    retargeted_motion_data["link_quat"] = torch.stack(retargeted_tracking_link_quat_global).numpy()
    retargeted_motion_data["foot_contact"] = torch.stack(foot_contact_list).numpy()

    return raw_motion_data, retargeted_motion_data

def load_smplx_data(smplx_file, body_models, target_fps):

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, body_models
    )

    smplx_data, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=target_fps
    )

    return smplx_data, fps, actual_human_height

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
    )

    parser.add_argument(
        "--save_dir",
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
    body_models = {
        "neutral": smplx.create(
            str(SMPLX_FOLDER),
            "smplx",
            gender="neutral",
            use_pca=False,
        ),
        "male": smplx.create(
            str(SMPLX_FOLDER),
            "smplx",
            gender="male",
            use_pca=False,
        ),
        "female": smplx.create(
            str(SMPLX_FOLDER),
            "smplx",
            gender="female",
            use_pca=False,
        ),
    }

    smplx_data, fps, actual_human_height = load_smplx_data(args.smplx_file, body_models, 30)

    env_args = EnvArgsRegistry["g1_motion"]
    envclass = getattr(envs, env_args.env_name)
    env = envclass(
        args=env_args,
        num_envs=1,
        show_viewer=args.view,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    raw_motion_data, retargeted_motion_data = retarget_smplx(smplx_data, fps, actual_human_height, env, args.view)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "raw_motion_data.pkl"), "wb") as f:
            pickle.dump(raw_motion_data, f)
        with open(os.path.join(args.save_dir, "retargeted_motion_data.pkl"), "wb") as f:
            pickle.dump(retargeted_motion_data, f)
        
    print(f"Saved motion data to {args.save_dir}")