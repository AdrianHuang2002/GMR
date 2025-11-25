
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
from .rot_utils import quatToEuler, flatten_quat_keep_yaw

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
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
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
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

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

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
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        # (A) optional: contact fixes in original human frame
        human_data = self.preprocess_contact_data(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

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

        # Optional: contact post-process on robot side
        qpos = self.configuration.data.qpos.copy()
        qpos = self.postprocess_robot_no_penetration(qpos)
        self.debug_robot_feet(qpos)
        # qpos = self.postprocess_robot_contact(qpos)
        return qpos

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
    
    def preprocess_contact_data_SE3(self, human_data): 
        human_feet_names = [k for k in human_data.keys() if "foot" in k]
        if len(human_feet_names) >= 2:
            left_name  = [n for n in human_feet_names if "left"  in n][0]
            right_name = [n for n in human_feet_names if "right" in n][0]

            left_pos,  left_quat  = human_data[left_name]
            right_pos, right_quat = human_data[right_name]

            foot_pos = np.array([left_pos, right_pos], dtype=float)
            foot_quat = np.array([left_quat, right_quat], dtype=float)
            foot_euler = np.array([quatToEuler(foot_quat[0]), quatToEuler(foot_quat[1]),], dtype=float)

            foot_tilt = np.clip((np.abs(foot_euler[:, 0]) + np.abs(foot_euler[:, 1]) - 0.4) / 0.4, 0.0, 1.0)
            foot_lift = np.clip((foot_pos[:, 2] - 0.15) / 0.15, 0.0, 1.0)

            if self.foot_last_pos is None:
                self.foot_last_pos = foot_pos.copy()

            foot_vel = np.clip(np.linalg.norm((foot_pos[..., :2] - self.foot_last_pos[..., :2]) / (1/30), axis=-1) - 0.15, 0.0, 1.0,)
            self.foot_last_pos = foot_pos.copy()

            foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clip(0.0, 1.0)
            foot_contact = 1.0 - foot_not_contact

            contact_thresh = 0.8

            if foot_contact[0] > contact_thresh:
                human_data = self.rigid_align_body_to_ground_SE3_v1(human_data, left_name, ground_height=self.ground[2])
            if foot_contact[1] > contact_thresh:
                human_data = self.rigid_align_body_to_ground_SE3_v1(human_data, right_name, ground_height=self.ground[2])
                
        return human_data
    
    def rigid_align_body_to_ground_SE3_v1(self, human_data, foot_name, ground_height=0.0):

        p_foot_orig, q_foot_orig = human_data[foot_name]
        R_foot_orig = R.from_quat(q_foot_orig, scalar_first=True).as_matrix()

        p_foot_des = np.array([p_foot_orig[0], p_foot_orig[1], ground_height])
        q_foot_des = flatten_quat_keep_yaw(q_foot_orig)  
        R_foot_des = R.from_quat(q_foot_des, scalar_first=True).as_matrix()

        R_delta = R_foot_des @ R_foot_orig.T
        p_delta = p_foot_des - R_delta @ p_foot_orig

        for name in human_data.keys():
            p_orig, q_orig = human_data[name]

            R_orig = R.from_quat(q_orig, scalar_first=True).as_matrix()

            p_new = R_delta @ p_orig + p_delta
            R_new = R_delta @ R_orig

            q_new = R.from_matrix(R_new).as_quat(scalar_first=True)

            human_data[name] = (p_new, q_new)

        return human_data

    
    def postprocess_robot_no_penetration(self, qpos):
        """
        Ensure no robot foot link is below ground.
        This does *not* change foot orientation, only shifts the whole robot up if needed.
        """
        data = mj.MjData(self.model)
        data.qpos[:] = qpos
        mj.mj_forward(self.model, data)

        left_id  = self.robot_body_names["left_toe_link"]
        right_id = self.robot_body_names["right_toe_link"]

        foot_pos = np.array([
            data.xpos[left_id],
            data.xpos[right_id],
        ], dtype=float)

        ground_z = self.ground[2]  # or 0.0 if your world ground is z=0
        min_z = foot_pos[:, 2].min()

        if min_z < ground_z:
            # shift entire robot up so the lowest foot is exactly on the ground
            qpos[2] += (ground_z - min_z)

        return qpos.copy()

    def debug_robot_feet(self, qpos):
        data = mj.MjData(self.model)
        data.qpos[:] = qpos
        mj.mj_forward(self.model, data)
        
        # Replace with your actual foot body names
        left_id  = self.robot_body_names["left_toe_link"]
        right_id = self.robot_body_names["right_toe_link"]
        
        print(
            "[DEBUG] robot foot z: L={:.4f}, R={:.4f}".format(
                data.xpos[left_id][2], data.xpos[right_id][2]
            )
        )

    def rigid_align_robot_to_ground_SE3(self, qpos, foot_body_name, ground_height=None):
        """
        Use the same rigid alignment idea as `rigid_align_body_to_ground`,
        but applied to the ROBOT BASE (qpos[0:7]) instead of human_data dict.

        - Takes the current robot state `qpos`.
        - Reads the given foot's world pose from MuJoCo.
        - Computes ΔT that:
            * keeps foot x,y
            * sets foot z = ground_height
            * flattens roll/pitch, keeps yaw (via flatten_quat_keep_yaw)
        - Applies that ΔT to the ROBOT BASE only.
        """
        if ground_height is None:
            ground_height = self.ground[2]

        # --- 1. Get current robot state in world ---
        data = mj.MjData(self.model)
        data.qpos[:] = qpos
        mj.mj_forward(self.model, data)

        foot_id = self.robot_body_names[foot_body_name]

        # MuJoCo stores xquat as [w,x,y,z]
        p_foot_orig = data.xpos[foot_id].copy()
        q_foot_orig = data.xquat[foot_id].copy()  # [w,x,y,z]

        R_foot_orig = R.from_quat(q_foot_orig, scalar_first=True).as_matrix()

        # --- 2. Desired grounded pose for that foot ---
        p_foot_des = np.array([p_foot_orig[0], p_foot_orig[1], ground_height])
        q_foot_des = flatten_quat_keep_yaw(q_foot_orig)       # [w,x,y,z]
        R_foot_des = R.from_quat(q_foot_des, scalar_first=True).as_matrix()

        # --- 3. Compute ΔT (same as your human version) ---
        R_delta = R_foot_des @ R_foot_orig.T
        p_delta = p_foot_des - R_delta @ p_foot_orig

        # --- 4. Apply ΔT to the ROBOT BASE pose only ---
        qpos_new = qpos.copy()

        # base pos & quat in MuJoCo qpos (assuming free base)
        base_pos  = qpos_new[0:3]
        base_quat = qpos_new[3:7]  # [w,x,y,z]

        R_base = R.from_quat(base_quat, scalar_first=True).as_matrix()

        base_pos_new  = R_delta @ base_pos + p_delta
        R_base_new    = R_delta @ R_base
        base_quat_new = R.from_matrix(R_base_new).as_quat(scalar_first=True)

        qpos_new[0:3] = base_pos_new
        qpos_new[3:7] = base_quat_new

        return qpos_new

    def postprocess_robot_contact_SE3(self, qpos, dt=1/30.0):
        data = mj.MjData(self.model)
        data.qpos[:] = qpos
        mj.mj_forward(self.model, data)

        left_name  = "left_toe_link"
        right_name = "right_toe_link"
        left_id  = self.robot_body_names[left_name]
        right_id = self.robot_body_names[right_name]

        foot_pos = np.array([data.xpos[left_id], data.xpos[right_id]], dtype=float)
        foot_quat = np.array([data.xquat[left_id], data.xquat[right_id]], dtype=float)  # [w,x,y,z]

        # tilt
        foot_euler = np.array([
            quatToEuler(foot_quat[0]),
            quatToEuler(foot_quat[1]),
        ], dtype=float)
        foot_tilt = np.clip(
            (np.abs(foot_euler[:, 0]) + np.abs(foot_euler[:, 1]) - 0.4) / 0.4,
            0.0, 1.0
        )

        # lift (relative to ground)
        ground_z = self.ground[2]
        foot_height = foot_pos[:, 2] - ground_z
        foot_lift = np.clip((foot_height - 0.15) / 0.15, 0.0, 1.0)

        # velocity
        if self.robot_foot_last_pos is None:
            self.robot_foot_last_pos = foot_pos.copy()
        foot_vel_xy = np.linalg.norm(
            (foot_pos[..., :2] - self.robot_foot_last_pos[..., :2]) / dt,
            axis=-1
        )
        self.robot_foot_last_pos = foot_pos.copy()
        foot_vel = np.clip(foot_vel_xy - 0.15, 0.0, 1.0)

        foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clip(0.0, 1.0)
        foot_contact = 1.0 - foot_not_contact

        contact_thresh = 0.8
        left_in_contact  = foot_contact[0] > contact_thresh
        right_in_contact = foot_contact[1] > contact_thresh

        # --- Use rigid alignment on *one* support foot only ---
        if left_in_contact and not right_in_contact:
            qpos = self.rigid_align_robot_to_ground_SE3(qpos, left_name, ground_height=ground_z)
        elif right_in_contact and not left_in_contact:
            qpos = self.rigid_align_robot_to_ground_SE3(qpos, right_name, ground_height=ground_z)
        else:
            # double support or no clear support: maybe just do a simple z-fix
            qpos = self.postprocess_robot_no_penetration(qpos)

        return qpos
    
    def preprocess_contact_data(self, human_data): 
        left_pos,  left_quat  = human_data["left_foot"]
        right_pos, right_quat = human_data["right_foot"]

        foot_pos = np.array([left_pos, right_pos], dtype=float)
        foot_quat = np.array([left_quat, right_quat], dtype=float)
        foot_euler = np.array([quatToEuler(foot_quat[0]), quatToEuler(foot_quat[1]),], dtype=float)
        foot_tilt = np.clip((np.abs(foot_euler[:, 0]) + np.abs(foot_euler[:, 1]) - 0.4) / 0.4, 0.0, 1.0)
        foot_lift = np.clip((foot_pos[:, 2] - 0.15) / 0.15, 0.0, 1.0)

        if self.foot_last_pos is None:
            self.foot_last_pos = foot_pos.copy()

        foot_vel = np.clip(np.linalg.norm((foot_pos[..., :2] - self.foot_last_pos[..., :2]) / (1/30.0), axis=-1) - 0.15, 0.0, 1.0,)
        self.foot_last_pos = foot_pos.copy()

        foot_not_contact = ((foot_tilt + foot_lift + foot_vel) / 1.5).clip(0.0, 1.0)
        foot_contact = 1.0 - foot_not_contact

        contact_thresh = 0.3

        if foot_contact[0] > contact_thresh:
            pos, quat = human_data["left_foot"]
            flat_quat = flatten_quat_keep_yaw(quat)       
            human_data["left_foot"] = (pos, flat_quat)
            human_data = self.offset_human_data_to_ground(human_data)

        if foot_contact[1] > contact_thresh:
            pos, quat = human_data["right_foot"]
            flat_quat = flatten_quat_keep_yaw(quat)
            human_data["right_foot"] = (pos, flat_quat)
            human_data = self.offset_human_data_to_ground(human_data)

        return human_data

   