
from xxlimited import foo
import mink
import mujoco as mj
import numpy as np
import json
import torch
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, sosfilt, sosfilt_zi
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
from .rot_utils import quatToEuler, flatten_quat_keep_yaw, slerp

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = 1.7,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        aligned_fps: float = 30,
        contact_filter: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        ratio = actual_human_height / ik_config["human_height_assumption"]
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.foot_last_pos = None
        self.foot_contact_list = []
        self.robot_foot_last_pos = None
        self.dt = 1.0 / aligned_fps

        self.contact_filter = contact_filter

        # Error tracking across frames (only final errors after IK iterations)
        self.all_final_errors_table1 = []
        self.all_final_errors_table2 = []

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        self.setup_retarget_configuration()

        self.ground_offset = 0.0
        self.human_to_robot_tracking = {
            # Core body
            "pelvis": "pelvis",
            "spine3": "torso_link",

            # Left leg
            # "left_hip": "left_hip_roll_link",
            # "left_knee": "left_knee_link",
            "left_foot": "left_ankle_roll_link",

            # Right leg
            # "right_hip": "right_hip_roll_link",
            # "right_knee": "right_knee_link",
            "right_foot": "right_ankle_roll_link",

            # Left arm
            # "left_shoulder": "left_shoulder_roll_link",
            # "left_elbow": "left_elbow_link",
            "left_wrist": "left_wrist_yaw_link",

            # Right arm
            # "right_shoulder": "right_shoulder_roll_link",
            # "right_elbow": "right_elbow_link",
            "right_wrist": "right_wrist_yaw_link",
        }


    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)

        if self.contact_filter:
            human_data = self.foot_contact_filter(human_data)
        else:
            self.foot_contact_data(human_data)

        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        
        import ipdb; ipdb.set_trace()
        self.scaled_human_data = human_data

    def retarget(self):

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = self.scaled_human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = self.scaled_human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1
            
            # Store per-frame final error (after all iterations)
            self.all_final_errors_table1.append(next_error)

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0

            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
            
            # Store per-frame final error (after all iterations)
            self.all_final_errors_table2.append(next_error)

        # self.debug_robot_feet(qpos)
        return self.configuration.data.qpos.copy()
    
    def calculate_error_statistics_and_plot(self, save_path=None):
        """Calculate final statistics for all collected errors and log them to a file."""
        import os
        
        all_stats = {}
        
        def calculate_stats(errors, name):
            """Calculate statistics for a list of errors."""
            if len(errors) == 0:
                return None
            
            errors_arr = np.array(errors)
            stats = {
                'min': float(np.min(errors_arr)),
                'max': float(np.max(errors_arr)),
                'mean': float(np.mean(errors_arr)),
                'std': float(np.std(errors_arr)),
                'percentiles': {
                    '25th': float(np.percentile(errors_arr, 25)),
                    '50th': float(np.percentile(errors_arr, 50)),
                    '75th': float(np.percentile(errors_arr, 75)),
                    '90th': float(np.percentile(errors_arr, 90)),
                    '95th': float(np.percentile(errors_arr, 95)),
                    '99th': float(np.percentile(errors_arr, 99)),
                }
            }
            print(f"\n[{name}] Final Statistics:")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Percentiles:")
            print(f"    25th: {stats['percentiles']['25th']:.6f}")
            print(f"    50th: {stats['percentiles']['50th']:.6f}")
            print(f"    75th: {stats['percentiles']['75th']:.6f}")
            print(f"    90th: {stats['percentiles']['90th']:.6f}")
            print(f"    95th: {stats['percentiles']['95th']:.6f}")
            print(f"    99th: {stats['percentiles']['99th']:.6f}")
            return stats
        
        # Calculate statistics for Table 1
        if self.use_ik_match_table1:
            if len(self.all_final_errors_table1) > 0:
                stats = calculate_stats(self.all_final_errors_table1, "IK Table 1 - final_error")
                if stats is not None:
                    all_stats["IK Table 1 - final_error"] = stats
        
        # Calculate statistics for Table 2
        if self.use_ik_match_table2:
            if len(self.all_final_errors_table2) > 0:
                stats = calculate_stats(self.all_final_errors_table2, "IK Table 2 - final_error")
                if stats is not None:
                    all_stats["IK Table 2 - final_error"] = stats
        
        # Log statistics to file
        if save_path and len(all_stats) > 0:
            # Prepare log file path
            base_path = save_path
            if base_path.endswith('.pkl'):
                base_path = base_path[:-4]
            log_path = base_path + '_error_stats.json'
            
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_path)
            if log_dir:  # Only create directory if it's not empty
                os.makedirs(log_dir, exist_ok=True)
            
            # Save as JSON
            with open(log_path, 'w') as f:
                json.dump(all_stats, f, indent=2)
            
            print(f"\n[Log] Statistics saved to: {log_path}")
    
    def reset_error_tracking(self):
        """Reset all error tracking lists."""
        self.all_final_errors_table1 = []
        self.all_final_errors_table2 = []

    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.0
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data

    def debug_robot_feet(self, qpos):
        """Check if feet are below ground for a given qpos."""
        # Build a fresh MjData and run FK with this qpos
        data = mj.MjData(self.model)
        data.qpos[:] = qpos
        mj.mj_forward(self.model, data)

        # Use BODY IDs, not DOF IDs
        left_id  = self.robot_body_names["left_toe_link"]   # or "left_ankle_roll_link"
        right_id = self.robot_body_names["right_toe_link"]  # or "right_ankle_roll_link"

        left_z  = data.xpos[left_id][2]
        right_z = data.xpos[right_id][2]
        ground_z = self.ground[2]  # usually 0.0

        if left_z < ground_z or right_z < ground_z:
            print("[WARNING] Robot foot penetration detected!")
            print(f"  left_z  = {left_z:.6f}")
            print(f"  right_z = {right_z:.6f}")
            print(f"  ground  = {ground_z:.6f}")
        # else:
        #     print(f"[OK] Feet above ground: L={left_z:.6f}, R={right_z:.6f}")

    def foot_contact_data(self, human_data):
        left_pos, left_quat = human_data["left_foot"]
        right_pos, right_quat = human_data["right_foot"]

        foot_pos = np.array([left_pos, right_pos], dtype=float)   # (2,3)
        foot_quat = np.array([left_quat, right_quat], dtype=float)  # (2,4)

        # --------- contact confidence ----------
        foot_euler = np.array(
            [quatToEuler(foot_quat[0]), quatToEuler(foot_quat[1])],
            dtype=float,
        )
        foot_tilt = np.clip((np.abs(foot_euler[:, 0]) + np.abs(foot_euler[:, 1]) - 0.4) / 0.4, 0.0, 1.0)
        foot_lift = np.clip((foot_pos[:, 2] - 0.15) / 0.15, 0.0, 1.0)

        if self.foot_last_pos is None:
            self.foot_last_pos = foot_pos.copy()

        foot_vel = np.clip(
            (np.linalg.norm((foot_pos[..., :2] - self.foot_last_pos[..., :2]) / self.dt, axis=-1) - 0.1) / 0.1,
            0.0, 1.0,
        )
        self.foot_last_pos = foot_pos.copy()

        foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clip(0.0, 1.0)
        foot_contact = 1.0 - foot_not_contact
        
        self.last_foot_contact = foot_contact
    
    def foot_contact_filter(self, human_data):
        """
        Support-foot rule (global z-shift once per frame) + per-foot rotation flatten (local).
        - Choose support foot = argmax(contact weight x)
        - Compute one ground_offset so support foot z -> ground_z (blended by support weight)
        - Apply this offset ONCE to all bodies (global)
        - Flatten each foot rotation (roll/pitch) using slerp with its own x[i] (local)
        """

        left_pos, left_quat = human_data["left_foot"]
        right_pos, right_quat = human_data["right_foot"]

        foot_pos = np.array([left_pos, right_pos], dtype=float)   # (2,3)
        foot_quat = np.array([left_quat, right_quat], dtype=float)  # (2,4)

        # --------- contact confidence ----------
        foot_euler = np.array(
            [quatToEuler(foot_quat[0]), quatToEuler(foot_quat[1])],
            dtype=float,
        )
        foot_tilt = np.clip((np.abs(foot_euler[:, 0]) + np.abs(foot_euler[:, 1]) - 0.4) / 0.4, 0.0, 1.0)
        foot_lift = np.clip((foot_pos[:, 2] - 0.15) / 0.15, 0.0, 1.0)

        if self.foot_last_pos is None:
            self.foot_last_pos = foot_pos.copy()

        foot_vel = np.clip(
            (np.linalg.norm((foot_pos[..., :2] - self.foot_last_pos[..., :2]) / self.dt, axis=-1) - 0.1) / 0.1,
            0.0, 1.0,
        )
        self.foot_last_pos = foot_pos.copy()

        foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clip(0.0, 1.0)
        foot_contact = 1.0 - foot_not_contact

        enter = 0.4
        full  = 0.9
        x_raw = np.clip((foot_contact - enter) / (full - enter), 0.0, 1.0)
        alpha = 0.3
        x = alpha * x_raw + (1 - alpha) * getattr(self, "last_foot_contact", x_raw)
        self.last_foot_contact = x

        # --------- Support-foot rule: compute ONE global z offset ----------
        ground_z = float(getattr(self, "ground", np.array([0.0, 0.0, 0.0]))[2])

        # pick support foot by confidence
        support_idx = int(np.argmax(x))
        w_sup = float(x[support_idx])
        z_sup = float(foot_pos[support_idx, 2])  

        dz_support = w_sup * (ground_z - z_sup)

        # anti-penetration shift: ensure lowest foot is not below ground
        z_min = float(np.min(foot_pos[:, 2]))
        dz_nopen = ground_z - z_min
        dz = max(dz_support, dz_nopen)
        # Apply the global offset ONCE
        self.set_ground_offset(-dz)
        human_data = self.apply_ground_offset(human_data)

        # --------- Per-foot rotation flatten (local) ----------
        # Keep positions as-is (already globally shifted), only adjust orientation with slerp

        # Left
        pos, quat = human_data["left_foot"]
        quat = np.asarray(quat, dtype=float)
        flat_quat = flatten_quat_keep_yaw(quat)
        wl = float(x[0])
        q_out = slerp(
            torch.tensor(quat, dtype=torch.float32),
            torch.tensor(flat_quat, dtype=torch.float32),
            torch.tensor(wl, dtype=torch.float32),
        ).detach().cpu().numpy()
        human_data["left_foot"] = (pos, q_out)

        # Right
        pos, quat = human_data["right_foot"]
        quat = np.asarray(quat, dtype=float)
        flat_quat = flatten_quat_keep_yaw(quat)
        wr = float(x[1])
        q_out = slerp(
            torch.tensor(quat, dtype=torch.float32),
            torch.tensor(flat_quat, dtype=torch.float32),
            torch.tensor(wr, dtype=torch.float32),
        ).detach().cpu().numpy()
        human_data["right_foot"] = (pos, q_out)

        return human_data
    
    def get_human_tracking_targets(self):
        """Use the IK task SE3 target directly for visualization."""
        targets = {}

        for human_name, task in self.human_body_to_task1.items():
            robot_link = self.human_to_robot_tracking.get(human_name, None)
            if robot_link is None:
                continue
            
            se3 = task.transform_target_to_world
            pos = np.array(se3.translation())
            quat = np.array(se3.rotation().wxyz)
            if robot_link == "left_ankle_roll_link":
                toe_id = self.robot_body_names["left_toe_link"]
                pos, quat = self.convert_child_to_parent_target(toe_id, pos, quat)

            if robot_link == "right_ankle_roll_link":
                toe_id = self.robot_body_names["right_toe_link"]
                pos, quat = self.convert_child_to_parent_target(toe_id, pos, quat)
            
            if robot_link == "torso_link":
                p_pelvis_w = self.human_body_to_task1["pelvis"].transform_target_to_world.translation()
                q_torso_w_wxyz = quat
                pos, quat = self.pelvis_to_torso_manual(p_pelvis_w, q_torso_w_wxyz)
                
            targets[robot_link] = (pos, quat)

        return targets
    
    def convert_child_to_parent_target(self, child_id, p_child_w, q_child_w_wxyz):
        # child pose relative to its parent (from MuJoCo model)
        p_child_in_parent = self.model.body_pos[child_id].copy()     # xyz in parent frame
        q_child_in_parent = self.model.body_quat[child_id].copy()    # wxyz in parent frame

        # rotations
        R_child_w = R.from_quat(q_child_w_wxyz, scalar_first=True)
        R_child_parent = R.from_quat(q_child_in_parent, scalar_first=True)

        R_parent_w = R_child_w * R_child_parent.inv()

        p_parent_w = p_child_w - R_parent_w.apply(p_child_in_parent)
        q_parent_w = R_parent_w.as_quat(scalar_first=True)

        return p_parent_w, q_parent_w
    
    def pelvis_to_torso_manual(self, p_pelvis_w, q_torso_w_wxyz):
        """
        Compute torso world position from pelvis world position.
        Torso orientation is assumed to be already known.
        """
        # fixed offset from pelvis frame (XML)
        p_offset = np.array([-0.0039635, 0.0, 0.044], dtype=float)

        R_pelvis = R.from_quat(q_torso_w_wxyz, scalar_first=True)
        p_torso_w = p_pelvis_w + R_pelvis.apply(p_offset)

        return p_torso_w, q_torso_w_wxyz
    
