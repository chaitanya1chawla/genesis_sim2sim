import os
import argparse
import time
import mujoco
import mujoco.viewer
from threading import Thread
from pathlib import Path
import threading
import numpy as np

import config_rep as config
import yaml
import numpy as np
import mujoco
import config_rep as config
import matplotlib.pyplot as plt


#######################################################
# Helper functions
#######################################################

def _print_joint_properties(mj_model):

    """Print joint properties similar to Genesis bimanual_robot.py"""
    print("\n" + "="*80)
    print("JOINT PROPERTIES")
    print("="*80)
    
    actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    
    for i in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name is None:
            joint_name = f"unnamed_joint_{i}"
        
        print(f"\nJoint {i}: {joint_name}")
        
        # Get joint DOF address
        dof_adr = mj_model.jnt_dofadr[i]
        
        # Check if joint has DOF (some joints like fixed joints don't)
        if dof_adr >= 0 and dof_adr < mj_model.nv:
            # Print joint limits
            if mj_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                if mj_model.jnt_range[i, 0] < mj_model.jnt_range[i, 1]:
                    print(f"  DOF limits: [{mj_model.jnt_range[i, 0]}, {mj_model.jnt_range[i, 1]}]")
            
            print(f"  DOF damping (set): {mj_model.dof_damping[dof_adr]}")
            print(f"  DOF armature (set): {mj_model.dof_armature[dof_adr]}")
            print(f"  DOF frictionloss (set): {mj_model.dof_frictionloss[dof_adr]}")
            
            # Find actuator for this joint (if exists)
            actuator_idx = None
            for act_idx in range(mj_model.nu):
                if mj_model.actuator_trnid[act_idx, 0] == i:
                    actuator_idx = act_idx
                    break
            
            if actuator_idx is not None:
                # Print kp (positional gain)
                kp = mj_model.actuator_gainprm[actuator_idx, 0]
                print(f"  DOF kp (set): {kp}")
                
                # Print kv (velocity gain) - stored as negative in biasprm[2]
                kv = -mj_model.actuator_biasprm[actuator_idx, 2] if mj_model.actuator_biasprm[actuator_idx, 2] != 0 else 0.0
                print(f"  DOF kv (set): {kv}")
                
                # Print force range
                lower_force = mj_model.actuator_forcerange[actuator_idx, 0]
                upper_force = mj_model.actuator_forcerange[actuator_idx, 1]
                print(f"  DOF force_range (set): [{lower_force}, {upper_force}]")
            else:
                print(f"  (No actuator found for this joint)")
        else:
            print(f"  (Fixed joint - no DOF)")
    
    print("="*80 + "\n")

# Set actuator and joint properties from config groups
def _set_actuator_and_joint_properties(mj_model, config_groups):
    """Set actuator gains, damping, force ranges, and joint properties from config groups."""
    actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    
    for group_name, group_cfg in config_groups.items():
        joint_exprs = group_cfg["joint_exprs"]
        kp_values = group_cfg.get("kp", None)
        kd_values = group_cfg.get("kd", None)
        force_range_values = group_cfg.get("force_range", None)
        damping_values = group_cfg.get("damping", None)
        armature_values = group_cfg.get("armature", None)
        frictionloss_values = group_cfg.get("frictionloss", None)

        print(f"\nSetting properties for {group_name}:")
        for idx, joint_expr in enumerate(joint_exprs):
            # Find actuator index for this joint
            actuator_idx = actuator_names.index(joint_expr)
            assert actuator_idx != -1, f"Actuator '{joint_expr}' not found"
            
            mj_model.actuator(actuator_idx).gainprm[0] = kp_values[idx]
            mj_model.actuator(actuator_idx).biasprm[1] = -kp_values[idx]  # Negative for damping
            mj_model.actuator(actuator_idx).biasprm[2] = -kd_values[idx]  # Negative for damping
            
            max_force = force_range_values[idx]
            mj_model.actuator(actuator_idx).forcerange[0] = -max_force
            mj_model.actuator(actuator_idx).forcerange[1] = max_force
            
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_expr)
            assert joint_id != -1, f"Joint '{joint_expr}' not found"
            dof_adr = mj_model.jnt_dofadr[joint_id]
            mj_model.dof_damping[dof_adr] = damping_values[idx] if damping_values \
                                                        else config.GENESIS_DEFAULTS["damping"]
            mj_model.dof_armature[dof_adr] = armature_values[idx] if armature_values \
                else config.GENESIS_DEFAULTS["armature"]
            mj_model.dof_frictionloss[dof_adr] = frictionloss_values[idx] if frictionloss_values \
                else config.GENESIS_DEFAULTS["frictionloss"]

def _print_joint_geom_body_info(mj_model):
    all_joint_names = []
    for i in range(mj_model.njnt):
        # mjtObj.mjOBJ_JOINT tells MuJoCo we are looking for joint names
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        all_joint_names.append(name)
        print(f"Joint ID {i}: {name}")
    
    # for i in range(mj_model.nbody):
    #     # mjtObj.mjOBJ_BODY tells MuJoCo we are looking for body names
    #     name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        
    #     # Body 0 is always "world", and unnamed bodies might return None or empty string
    #     if name is None:
    #         name = "world" if i == 0 else f"unnamed_body_{i}"
    #     print(f"Body ID {i}: {name}")


#######################################################
# Control functions
#######################################################

def _compute_pd_torques(mj_model, mj_data, target_positions, kp_per_actuator, kd_per_actuator):
    """
    Compute PD control torques: tau = Kp * (q_ref - qpos) - Kd * qvel.

    Used when actuators are motors (ctrl = torque). For position actuators,
    MuJoCo applies this internally: you set ctrl = target position and
    the actuator uses kp/kv from gainprm/biasprm.

    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data (must have been through mj_forward for correct qpos/qvel)
        target_positions: array of length mj_model.nu, target position per actuator (rad)
        kp_per_actuator: array of length mj_model.nu, position gain per actuator
        kd_per_actuator: array of length mj_model.nu, velocity gain per actuator

    Returns:
        torques: array of length mj_model.nu (Nm), clamped to actuator forcerange
    """
    torques = np.zeros(mj_model.nu)
    for i in range(mj_model.nu):
        joint_id = mj_model.actuator_trnid[i, 0]
        if joint_id < 0:
            continue

        dof_adr = mj_model.jnt_dofadr[joint_id]
        qpos_adr = mj_model.jnt_qposadr[joint_id]

        # Calculate PD torque
        q_ref = target_positions[i]
        tau = kp_per_actuator[i] * (q_ref - mj_data.qpos[qpos_adr]) \
            - kd_per_actuator[i] * (mj_data.qvel[dof_adr])

        # Set torque limits
        lo, hi = mj_model.actuator_forcerange[i, 0], mj_model.actuator_forcerange[i, 1]
        torques[i] = np.clip(tau, lo, hi)
    return torques


def _get_actuator_kp_kd(mj_model, config_groups):
    """
    Build kp and kd arrays (length mj_model.nu) from config groups, in actuator order.
    Used by compute_pd_torques when using motor actuators.
    """
    actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
    kp = np.zeros(mj_model.nu)
    kd = np.zeros(mj_model.nu)
    for group_cfg in config_groups.values():
        for idx, joint_expr in enumerate(group_cfg["joint_exprs"]):
            actuator_idx = actuator_names.index(joint_expr)
            kp[actuator_idx] = group_cfg["kp"][idx]
            kd[actuator_idx] = group_cfg["kd"][idx]
    return kp, kd


def _compute_gravity_compensation(mj_model, mj_data, compensation_scale=1.0, return_torque=True):
    """
    Compute gravity compensation per actuator.

    For affine/position actuators (return_torque=False): returns position offset
    (tau_g / kp) so that ctrl = target + offset produces torque tau_g.

    For motor actuators (return_torque=True): returns tau_g per actuator (Nm)
    to add to PD torque.
    """
    mujoco.mj_forward(mj_model, mj_data)
    
    actuator_offsets = np.zeros(mj_model.nu)
    
    for i in range(mj_model.nu):
        joint_id = mj_model.actuator_trnid[i, 0]
        if joint_id < 0:
            continue
        dof_adr = mj_model.jnt_dofadr[joint_id]
        tau_g = mj_data.qfrc_bias[dof_adr]

        if return_torque:
            actuator_offsets[i] = tau_g * compensation_scale
        else:
            kp = mj_model.actuator_gainprm[i, 0]
            if kp > 1e-6:
                actuator_offsets[i] = (tau_g / kp) * compensation_scale
    return actuator_offsets


#######################################################
# Visualization functions
#######################################################

def _get_reference_joint_positions(mj_model, TRAJ, qpos_map, ch_g1_action_names, ch_g1_action_indices, ch_hand_action_names, ch_hand_action_indices, t):
    """Calculates world positions for joints at trajectory time t."""
    # 1. Set the reference qpos from the trajectory
    # Using 'retarget_data' which matches the 53-DOF genesis order
    qpos = np.zeros(53) # 53 dof

    for i, name in enumerate(ch_g1_action_names):
        qpos[qpos_map[name]] = TRAJ["qpos"][t][ch_g1_action_indices[i]]
    for i, name in enumerate(ch_hand_action_names):
        qpos[qpos_map[name]] = TRAJ["qpos"][t][ch_hand_action_indices[i]]

    ref_data = mujoco.MjData(mj_model)
    ref_data.qpos[:] = qpos.copy()
    
    # 2. Run Forward Kinematics
    # mj_kinematics only computes positions (faster than mj_forward)
    mujoco.mj_kinematics(mj_model, ref_data)
    
    return ref_data.xanchor

def _add_debug_sphere(viewer, position, radius=0.02, rgba=[1, 0, 0, 1]):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=position,
        mat=np.eye(3).flatten(),
        rgba=rgba
    )
    scene.ngeom += 1

def _plot_joints(actions, qpos, plot_joints, joint_names=None, title_prefix=""):
    """
    Plot actions and qpos for selected joints on the same figure,
    using one subplot per joint index in `plot_joints`.

    Expected shapes:
      actions: (T, J) or (T, J_total)  – already sliced to the joints’ space
      qpos:    (T, J) or (T, J_total)
    `plot_joints` is a list of integer joint indices into the second dim.
    """
    assert actions.shape[0] == qpos.shape[0], "time dimension mismatch"
    T = actions.shape[0]
    t = np.arange(T)

    n = len(plot_joints)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, j_idx in zip(axes, plot_joints):
        a = actions[:, j_idx]
        q = qpos[:, j_idx]

        name = f"joint {j_idx}"
        if joint_names is not None and j_idx < len(joint_names):
            name = joint_names[j_idx]

        ax.plot(t, a, label=f"action ({name})", color="C0")
        ax.plot(t, q, label=f"qpos ({name})", color="C1")
        ax.set_ylabel("value")
        ax.set_title(f"{title_prefix}{name}")
        ax.legend()

    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument("--onnx_suffix", type=str, default=None)
parser.add_argument("--obj_init_q", type=float, default=1.4036)
parser.add_argument("--genesis_ctrl", action="store_true")

args = parser.parse_args()

TRAJ_FILE = os.path.join(os.path.dirname(__file__), config.TRAJ_FILE)
data = np.load(TRAJ_FILE, allow_pickle=True)

TRAJ = {}
TRAJ["qpos"] = data["qpos"].squeeze(1)
TRAJ["actions"] = data["actions"].squeeze(1)

s = 0
locker = threading.Lock()

# Scene: affine = general actuators (position targets); motor = motor actuators (PD torques)
ACTUATOR_MODE = getattr(config, "ACTUATOR_MODE", "affine")
if ACTUATOR_MODE == "affine":
    robot_scene_path = getattr(config, "ROBOT_SCENE_AFFINE", config.ROBOT_SCENE.replace("scene_g1_29dof.xml", "scene_g1_29dof_affine.xml"))
else:
    robot_scene_path = config.ROBOT_SCENE
spec = mujoco.MjSpec.from_file(str(Path(__file__).parent / robot_scene_path))
print(f"actuator mode: {ACTUATOR_MODE}, robot scene: {str(Path(__file__).parent / robot_scene_path)}")


mj_model = spec.compile()
mj_data = mujoco.MjData(mj_model)


# Set solver and contact properties 
# default integrator is Euler.
mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
mj_model.opt.impratio = 10.0


_set_actuator_and_joint_properties(mj_model, config_groups = {
                                                "WAIST_GROUP": config.WAIST_GROUP, 
                                                "ARM_GROUP": config.ARM_GROUP, 
                                                "FINGER_GROUP": config.FINGER_GROUP,
                                            })
print("Actuator and joint properties set from config groups")
_print_joint_properties(mj_model)

actuator_names = [mj_model.actuator(i).name for i in range(mj_model.nu)]
print(actuator_names, len(actuator_names), mj_data.ctrl.shape)
num_policy_waist = 1
num_policy_arm = 14
num_policy_hand = 24
total_policy_dofs = num_policy_waist + num_policy_arm + num_policy_hand
# ch_joint_names = ['waist_yaw_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
# policy_to_robot_body_map = []
# for j in ch_joint_names:
#     policy_to_robot_body_map.append(actuator_names.index(j))

ch_g1_action_indices = config.CH_G1_ACTION_INDICES
ch_hand_action_indices = config.CH_HAND_ACTION_INDICES
uncontrolled_indices = config.UNCONTROLLED_INDICES
ch_joint_names = config.CH_JOINT_NAMES # 53 dof joint names in genesis order


ch_g1_action_names = ch_joint_names[ch_g1_action_indices]
ch_hand_action_names = ch_joint_names[ch_hand_action_indices]
ch_uncontrolled_names = ch_joint_names[uncontrolled_indices]

qpos_map = {}
for i in range(mj_model.njnt):
    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_adr = mj_model.jnt_qposadr[i]  # starting index in qpos
    qpos_size = mj_model.jnt_qposadr[i+1] - mj_model.jnt_qposadr[i] if i+1 < mj_model.njnt else mj_model.nq - mj_model.jnt_qposadr[i]
    qpos_map[joint_name] = qpos_adr

# genesis name -> mujoco idx
ctrl_g1_body_map = []
for j in ch_g1_action_names:
    ctrl_g1_body_map.append(actuator_names.index(j))
ctrl_hand_body_map = []
for j in ch_hand_action_names:
    ctrl_hand_body_map.append(actuator_names.index(j))
ctrl_uncontrolled_body_map = []
for j in ch_uncontrolled_names:
    ctrl_uncontrolled_body_map.append(actuator_names.index(j))



def set_qpos(mj_data, t):
    for i in range(len(ch_g1_action_names)):
        start = qpos_map[ch_g1_action_names[i]]
        mj_data.qpos[start] = TRAJ["qpos"][t][ch_g1_action_indices[i]]
    for i in range(len(ch_hand_action_names)):
        start_h = qpos_map[ch_hand_action_names[i]]
        mj_data.qpos[start_h] = TRAJ["qpos"][t][ch_hand_action_indices[i]]
    mujoco.mj_forward(mj_model, mj_data)

def set_zero_qpos(mj_data, idx):
    for i in idx:
        mj_data.qpos[i] = 0.0
        mj_data.qvel[i] = 0.0

M = 0
set_qpos(mj_data, M)
initial_free_pose = np.array(mj_data.qpos[:7])


viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# Customize the viewer options
# viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

def SimulationThread():
    global mj_data, mj_model, prev_upper_targets, initial_free_pose, buffers, s

    s = 0
    substeps = config.SUBSTEPS

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        # ================================================================
        # STEP 3: Translate actions to joint targets
        # ================================================================
        # Initialize control array to match mj_data.ctrl size (actuator space)
        full_ctrl = np.zeros(mj_model.nu)
        full_ctrl[ctrl_g1_body_map] = TRAJ["actions"][s][ch_g1_action_indices]
        # ================================================================
        # STEP 4: Affine vs motor actuators
        # affine: ctrl = position targets; MuJoCo applies gainprm/biasprm.
        # motor:  ctrl = PD torques from compute_pd_torques.
        # ================================================================
        if ACTUATOR_MODE == "motor":
            from pdb import set_trace as st
            st()
            mujoco.mj_forward(mj_model, mj_data)
            kp_arr, kd_arr = _get_actuator_kp_kd(mj_model, {
                "WAIST_GROUP": config.WAIST_GROUP,
                "ARM_GROUP": config.ARM_GROUP,
                "FINGER_GROUP": config.FINGER_GROUP,
            })
            full_ctrl = _compute_pd_torques(
                mj_model, mj_data, full_ctrl, kp_arr, kd_arr
            )
        
        # ================================================================
        # STEP 5: Gravity compensation
        # affine: position offset (tau_g/kp). motor: torque (tau_g).
        # ================================================================
        gravity_comp = np.zeros(mj_model.nu)
        if config.ENABLE_GRAVITY_COMPENSATION:
            gravity_comp = _compute_gravity_compensation(
                mj_model, mj_data,
                compensation_scale=config.GRAVITY_COMPENSATION_SCALE,
                return_torque=(ACTUATOR_MODE == "motor"),
            )
        full_ctrl += gravity_comp
        # ================================================================
        # STEP 6: Apply control to MuJoCo
        # ================================================================
        mj_data.ctrl[:] = full_ctrl
        s = min(s + 1, config.MAX_EP_LENGTH - 1)

        # ================================================================
        # STEP 2: SIMULATE PHYSICS (Substepping)
        # ================================================================
        # We hold the policy 'ctrl' constant while we step physics 6 times
        for _ in range(substeps):
            # set_zero_qpos(mj_data, ctrl_uncontrolled_body_map)
            # set_qpos(mj_data, s)
            mujoco.mj_step(mj_model, mj_data)

        # ================================================================
        # STEP 3: REAL-TIME SLEEP (Based on 30Hz control cycle)
        # ================================================================
        # Sleep until the next 30Hz cycle (0.0333s total duration)
        # because we performed 6 steps of config.SIMULATE_DT
        elapsed = time.perf_counter() - step_start
        target_period = mj_model.opt.timestep * substeps
        if elapsed < target_period:
            time.sleep(target_period - elapsed)
            
        locker.release()

        if s == config.MAX_EP_LENGTH:
            _plot_joints(TRAJ["actions"], mj_data.qpos, ch_g1_action_indices, joint_names=ch_g1_action_names, title_prefix="g1")


def PhysicsViewerThread():
    global s # Use the current episode step from SimulationThread
    
    while viewer.is_running():
        locker.acquire()
        
        # Reset debug geom counter
        viewer.user_scn.ngeom = 0
        
        # Calculate Reference positions for the current step 's'
        if TRAJ is not None and s < len(TRAJ["qpos"]):
            # Run FK for the reference trajectory frame
            ref_anchors = _get_reference_joint_positions(mj_model, TRAJ, qpos_map, ch_g1_action_names, ch_g1_action_indices, ch_hand_action_names, ch_hand_action_indices, s)
            
            # Visualize Reference Arms (Semi-transparent Red)
            for joint_name in ch_g1_action_names:
                jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid != -1:
                    pos = ref_anchors[jid]
                    _add_debug_sphere(viewer, pos, radius=0.02, rgba=[1, 0, 0, 0.5])

            # Visualize Reference Hands (Semi-transparent White)
            for joint_name in ch_hand_action_names:
                jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid != -1:
                    pos = ref_anchors[jid]
                    _add_debug_sphere(viewer, pos, radius=0.01, rgba=[1, 1, 1, 0.3])
                    
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()



# RUN examples:

# 1. For tracking:
#  python3 unitree_mujoco.py --obj_init_q 0                                                                          130 ↵

