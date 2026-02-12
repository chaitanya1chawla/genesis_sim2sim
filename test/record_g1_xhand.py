import argparse
import re
from pathlib import Path

import numpy as np
import torch
import genesis as gs
from omegaconf import OmegaConf
import matplotlib.pyplot as plt


def _get_actuated_joints(entity):
    """Return {joint_name: joint} for all revolute/prismatic joints."""
    return {
        joint.name: joint
        for joint in entity.joints
        if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
    }


def _find_joint_groups(joint_exprs, actuated_joints):
    """Mimic BimanualRobot.find_joint_groups: regex over joint names."""
    if not joint_exprs:
        return []
    joints = []
    for expr in joint_exprs:
        pattern = re.compile(expr)
        for joint in actuated_joints.values():
            if pattern.match(joint.name):
                joints.append(joint)
    return joints


def _set_group_joint_gains(entity, actuator_cfg, actuated_joints, device):
    """Apply kp/kv/force_range/armature/damping to a joint group."""
    joints = _find_joint_groups(actuator_cfg.joint_exprs, actuated_joints)
    if len(joints) == 0:
        return

    joint_idxs = [int(j.dof_idx_local) for j in joints]
    n_joints = len(joint_idxs)

    for key in ["kp", "kv", "force_range", "armature", "damping"]:
        value = actuator_cfg.get(key, None)
        if value is None:
            continue
        if isinstance(value, (float, int)):
            batched = torch.full((n_joints,), float(value), device=device, dtype=torch.float32)
        else:
            batched = torch.tensor(value, device=device, dtype=torch.float32)
        if key == "kp":
            entity.set_dofs_kp(batched, dofs_idx_local=joint_idxs)
        elif key == "kv":
            entity.set_dofs_kv(batched, dofs_idx_local=joint_idxs)
        elif key == "force_range":
            entity.set_dofs_force_range(-1.0 * batched, batched, dofs_idx_local=joint_idxs)
        elif key == "armature":
            entity.set_dofs_armature(batched, dofs_idx_local=joint_idxs)
        elif key == "damping":
            entity.set_dofs_damping(batched, dofs_idx_local=joint_idxs)


def _apply_all_joint_gains(entity, robot_cfg, actuated_joints, print_joint_gains, device):
    """Apply all actuator configs from robot_cfg.actuators."""
    for actuator_cfg in robot_cfg.actuators:
        _set_group_joint_gains(entity, actuator_cfg, actuated_joints, device)

    if print_joint_gains:

        for i, joint in enumerate(entity.joints):
            print(f"\nJoint {i}: {joint.name}")
            if hasattr(joint, 'dofs_limit'):
                print(f"  DOF limits: {joint.dofs_limit}")

            # Get actual set values from entity (not default values from joint)
            dof_idx = joint.dof_idx_local
            if dof_idx is not None:
                # Position limits
                lower_limit, upper_limit = entity.get_dofs_limit(dofs_idx_local=[dof_idx])
                print(f"  DOF limits (set): [{lower_limit[0].item()}, {upper_limit[0].item()}]")
                actual_damping = entity.get_dofs_damping(dofs_idx_local=[dof_idx])
                print(f"  DOF damping (set): {actual_damping[0].item()}")
                actual_armature = entity.get_dofs_armature(dofs_idx_local=[dof_idx])
                print(f"  DOF armature (set): {actual_armature[0].item()}")
                actual_kp = entity.get_dofs_kp(dofs_idx_local=[dof_idx])
                print(f"  DOF kp (set): {actual_kp[0].item()}")
                actual_kv = entity.get_dofs_kv(dofs_idx_local=[dof_idx])
                print(f"  DOF kv (set): {actual_kv[0].item()}")
                lower_force, upper_force = entity.get_dofs_force_range(dofs_idx_local=[dof_idx])
                print(f"  DOF force_range (set): [{lower_force[0].item()}, {upper_force[0].item()}]")
                actual_stiffness = entity.get_dofs_stiffness(dofs_idx_local=[dof_idx])
                print(f"  DOF stiffness (set): {actual_stiffness[0].item()}")
                invweight = entity.get_dofs_invweight(dofs_idx_local=[dof_idx])
                print(f"  DOF invweight (set): {invweight[0].item()}")
                frictionloss = entity.get_dofs_frictionloss(dofs_idx_local=[dof_idx])
                print(f"  DOF frictionloss (set): {frictionloss[0].item()}")


def build_scene_and_robot(
    env_cfg_path: Path,
    robot_cfg_path: Path,
    num_envs: int,
    show_viewer: bool,
    print_joint_gains: bool,
):
    """
    Minimal Genesis setup using env/robot YAMLs:
    - scene props from `env_cfg/default.yaml`
    - robot props from `g1_xhand.yaml`
    """
    env_cfg = OmegaConf.load(str(env_cfg_path))
    robot_cfg = OmegaConf.load(str(robot_cfg_path))

    # Allow CLI to override num_envs / viewer
    env_cfg.num_envs = num_envs
    env_cfg.scene_kwargs.show_viewer = show_viewer

    dt = 1.0 / float(env_cfg.freq)
    scene_kwargs = env_cfg.scene_kwargs

    sim_options = gs.options.SimOptions(
        dt=dt,
        substeps=int(scene_kwargs.substeps),
        gravity=(0.0, 0.0, -9.81),
    )

    rigid_options = gs.options.RigidOptions(
        dt=dt,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=bool(scene_kwargs.group_collisions),
        enable_joint_limit=bool(scene_kwargs.enable_joint_limit),
        batch_dofs_info=bool(scene_kwargs.batch_dofs_info),
    )

    vis_options = gs.options.VisOptions(
        n_rendered_envs=None
        if env_cfg.num_envs <= 0
        else (env_cfg.num_envs if scene_kwargs.n_rendered_envs is None else scene_kwargs.n_rendered_envs),
        show_world_frame=False,
        segmentation_level="entity",
    )

    # Use the "front" camera from env cfg for viewer defaults
    cam_front = env_cfg.camera_kwargs.front
    viewer_options = gs.options.ViewerOptions(
        camera_pos=tuple(cam_front.pos),
        camera_lookat=tuple(cam_front.lookat),
        camera_fov=float(cam_front.fov),
    )

    scene = gs.Scene(
        sim_options=sim_options,
        rigid_options=rigid_options,
        vis_options=vis_options,
        viewer_options=viewer_options,
        show_viewer=bool(scene_kwargs.show_viewer),
    )

    # Table and ground from env cfg
    scene.add_entity(
        gs.morphs.Box(
            pos=tuple(env_cfg.table_pos),
            size=tuple(env_cfg.table_size),
            fixed=True,
            visualization=True,
        ),
        surface=gs.surfaces.Default(color=(0.35, 0.2, 0.1, 1.0)),
    )

    scene.add_entity(
        gs.morphs.URDF(
            file=env_cfg.plane_urdf_path,
            fixed=True,
        )
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------------
    # Minimal robot loading, using only what we need from g1_xhand.yaml
    # ----------------------------------------------------------------------
    load_kwargs = robot_cfg.load_kwargs
    robot_entity = scene.add_entity(
        gs.morphs.URDF(
            file=load_kwargs.file,
            pos=tuple(load_kwargs.pos),
            quat=tuple(load_kwargs.quat),
            fixed=bool(load_kwargs.fixed),
            merge_fixed_links=bool(load_kwargs.merge_fixed_links),
            collision=bool(load_kwargs.collision),
            convexify=bool(load_kwargs.convexify),
        ),
        material=gs.materials.Rigid(
            gravity_compensation=load_kwargs.gravity_compensation),
        visualize_contact=bool(load_kwargs.visualize_contact),
    )

    scene.build(
        n_envs=env_cfg.num_envs,
        env_spacing=tuple(env_cfg.env_spacing),
        n_envs_per_row=env_cfg.n_envs_per_row,
    )
    scene.reset()


    # ------------------------------------------------------------------
    # Apply joint gains / PD parameters and set initial configuration
    # similar to BimanualRobot.post_scene_build_setup + reset()
    # ------------------------------------------------------------------
    actuated_joints = _get_actuated_joints(robot_entity)
    _apply_all_joint_gains(robot_entity, robot_cfg, actuated_joints, print_joint_gains, device)
    print("Done applying joint gains")

    # init_qpos = torch.tensor(robot_cfg.default_qpos, device=device, dtype=torch.float32)
    init_qpos = torch.zeros(len(actuated_joints), device=device, dtype=torch.float32)
    actuated_dof_idxs = [int(j.dof_idx_local) for j in actuated_joints.values()]    

    robot_entity.set_dofs_position(
        position=init_qpos,
        dofs_idx_local=actuated_dof_idxs,
        zero_velocity=True,
        envs_idx=None,
    )
    robot_entity.zero_all_dofs_velocity(envs_idx=None)

    # Precompute DOF indices and limits for simple joint control
    dof_idxs = actuated_dof_idxs
    lower, upper = robot_entity.get_dofs_limit(dofs_idx_local=dof_idxs)
    lower = torch.as_tensor(lower, device=device, dtype=torch.float32)
    upper = torch.as_tensor(upper, device=device, dtype=torch.float32)

    return scene, robot_entity, env_cfg, device, dof_idxs, lower, upper


def record_rollout(
    scene,
    robot_entity,
    env_cfg,
    device,
    dof_idxs,
    lower_limits,
    upper_limits,
    num_steps: int,
    out_path: Path,
):
    """
    Run a rollout with random actions and record:
    - actions: (T, B, action_dim)
    - qpos: (T, B, ndof)
    """
    actions_hist = []
    qpos_hist = []

    num_envs = int(env_cfg.num_envs)
    T = num_steps
    ndof = len(dof_idxs)

    for t in range(T):
        with torch.no_grad():
            # Sample actions in [-1, 1] and map to joint limits
            # actions = torch.rand(num_envs, ndof, device=device) * 2.0 - 1.0
            # q_targets = lower_limits + (actions + 1.0) * 0.5 * (upper_limits - lower_limits)

            sin_wave = torch.sin(torch.tensor(t, device=device) * 5.0 * torch.pi / T)
            sin_actions = sin_wave * 0.5


        ctrl_indices = [21, 22, 23, 24, 27, 28]

        q_targets = torch.zeros(num_envs, ndof, device=device)
        q_targets[:, ctrl_indices] = sin_actions
        
        qpos = robot_entity.get_dofs_position(dofs_idx_local=dof_idxs, envs_idx=None)
        qpos_hist.append(torch.as_tensor(qpos).cpu().numpy())
        actions_hist.append(q_targets.detach().cpu().numpy())

        robot_entity.control_dofs_position(
            q_targets,
            dofs_idx_local=dof_idxs,
            envs_idx=None,
        )
        scene.step()

    actions_arr = np.stack(actions_hist, axis=0)  # (T, B, ndof)
    qpos_arr = np.stack(qpos_hist, axis=0)        # (T, B, ndof)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        actions=actions_arr,
        qpos=qpos_arr,
    )
    plot_rollout(actions_arr, qpos_arr, env_idx=0, dof_idx=ctrl_indices)
    print(f"Saved rollout to {out_path} with shape actions={actions_arr.shape}, qpos={qpos_arr.shape}")


def plot_rollout(actions, qpos, env_idx: int = 0, dof_idx: list[int] = None):
    """
    Plot action and qpos for a list of DOFs of a single env vs time,
    using one subplot per DOF.
    """
    if dof_idx is None or len(dof_idx) == 0:
        return

    T = actions.shape[0]
    t = np.arange(T)

    # actions: (T, B, ndof), qpos: (T, B, ndof)
    a_env = actions[:, env_idx, :]  # (T, ndof)
    q_env = qpos[:, env_idx, :]     # (T, ndof)

    n = len(dof_idx)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, dof_idx):
        a = a_env[:, idx]
        q = q_env[:, idx]

        ax.plot(t, a, label=f"action (dof {idx})", color="C0")
        ax.plot(t, q, label=f"qpos (dof {idx})", color="C1")
        ax.set_ylabel("value")
        ax.set_title(f"Env {env_idx}, DOF {idx}")
        ax.legend()

    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.savefig(f"rollout_env_{env_idx}_dofs_{'_'.join(map(str, dof_idx))}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Launch Genesis with g1_xhand and record states/actions."
    )
    parser.add_argument(
        "--env-cfg",
        type=str,
        default="test/env_cfg_reproduce.yaml",
        help="Path to env config YAML (scene props).",
    )
    parser.add_argument(
        "--robot-cfg",
        type=str,
        default="test/g1_xhand_reproduce.yaml",
        help="Path to robot config YAML.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="test/g1_xhand_rollout.npz",
        help="Output .npz path for recorded data.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show Genesis viewer while running.",
    )
    parser.add_argument(
        "--print_joint_gains",
        action="store_true",
        help="Print joint gains after setting them.",
    )
    parser.add_argument(
        "--plot-dof-idx",
        type=int,
        default=0,
        help="DOF index to plot.",
    )

    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    env_cfg_path = Path(args.env_cfg)
    robot_cfg_path = Path(args.robot_cfg)
    out_path = Path(args.out)

    scene, robot_entity, env_cfg, device, dof_idxs, lower, upper = build_scene_and_robot(
        env_cfg_path=env_cfg_path,
        robot_cfg_path=robot_cfg_path,
        num_envs=args.num_envs,
        show_viewer=args.viewer,
        print_joint_gains=args.print_joint_gains,
    )

    steps = int(env_cfg.episode_length)

    record_rollout(
        scene=scene,
        robot_entity=robot_entity,
        env_cfg=env_cfg,
        device=device,
        dof_idxs=dof_idxs,
        lower_limits=lower,
        upper_limits=upper,
        num_steps=steps,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()

