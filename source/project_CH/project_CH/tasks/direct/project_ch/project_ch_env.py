# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project CH environment - inherits from go2_piper_master."""

from __future__ import annotations

import torch
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from .project_ch_env_cfg import ProjectChEnvCfg


# Import master environment and config
try:
    from go2_piper_master.tasks.direct.go2_piper_master.go2_piper_master_env import Go2PiperMasterEnv
    from .project_ch_env_cfg import ProjectChEnvCfg
    MASTER_AVAILABLE = True
except ImportError:
    print("[WARNING] go2_piper_master not found. Using minimal fallback.")
    from isaaclab.envs import DirectRLEnv
    from .project_ch_env_cfg import ProjectChEnvCfg
    Go2PiperMasterEnv = DirectRLEnv
    MASTER_AVAILABLE = False


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class ProjectChEnv(Go2PiperMasterEnv):
    """Project CH environment (Direct Sytle) - extends master environment."""
  
    cfg: ProjectChEnvCfg

    def __init__(self, cfg: ProjectChEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize Project CH environment."""
        super().__init__(cfg, render_mode, **kwargs)

        if MASTER_AVAILABLE:
            print(f"[INFO] Project CH initialized with {cfg.scene.num_envs} environments")
        else:
            print("[WARNING] Running with fallback environment")
        
        # Locomotion-related states
        self.joint_gears = torch.ones(self.cfg.joint_gears, device=self.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears)
        
        self.potentials = torch.zeros(self.num_envs, device=self.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        
        self.targets = torch.tensor([1000, 0 , 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        self.targets = self.targets + self.scene.env_origins.to(dtype=torch.float32)
        
        self.start_rotation = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        
        self.basis_vec0 = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))
        self.basis_vec1 = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).repeat((self.num_envs, 1))

    # Apply action to the robot
    def _apply_action(self):
        forces = self.cfg.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_rewards(self) -> torch.Tensor:
        if not MASTER_AVAILABLE:
            return torch.zeros(self.num_envs, device=self.device)
        self._compute_intermediate_values()
        total_reward, reward_terms = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.heading_proj,
            self.up_proj,
            self.vel_loc,
            self.angvel_loc,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.motor_effort_ratio,

            # cfg에서 가져온 weight들
            self.cfg.heading_weight,
            self.cfg.alive_reward_scale,
            self.cfg.death_cost,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.track_lin_vel_xy_exp_weight,
            self.cfg.track_ang_vel_z_exp_weight,
            self.cfg.dof_torques_l2_weight,
            self.cfg.dof_acc_l2_weight,
            self.cfg.flat_orientation_l2_weight,
            self.cfg.feet_air_time_weight,
        )
        self.extras["episode"] = reward_terms
        return total_reward

    def _get_observations(self) -> dict:
        """CH-specific observations.""" 
        if not MASTER_AVAILABLE:
            return {"policy": torch.zeros((self.num_envs, 20), device=self.device)}
        self._compute_intermediate_values()
        obs = torch.cat(
            (
                self.torso_position[:, 2].unsqueeze(-1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

# new reward function 0728:17:17
@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    vel_loc: torch.Tensor,
    angvel_loc: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    motor_effort_ratio: torch.Tensor,

    # cfg에서 가져온 값들
    heading_weight: float,
    alive_reward_scale: float,
    death_cost: float,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    track_lin_vel_xy_exp_weight: float,
    track_ang_vel_z_exp_weight: float,
    dof_torques_l2_weight: float,
    dof_acc_l2_weight: float,
    feet_air_time_weight: float,
    flat_orientation_l2_weight: float,
):
    # 1. 기본 보상 (progress + alive)
    progress_reward = potentials - prev_potentials
    alive_reward = torch.ones_like(progress_reward) * alive_reward_scale

    # 2. Heading 보상
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(
        heading_proj > 0.8,
        heading_weight_tensor,
        heading_weight * heading_proj / 0.8,
    )

    # 3. Up 방향 보정 (기울기 보상/패널티)
    orientation_penalty = torch.where(
        up_proj < 0.93,
        torch.ones_like(up_proj) * flat_orientation_l2_weight,
        torch.zeros_like(up_proj),
    )

    # 4. 전진 속도 보상 (목표 속도 = 1 m/s 가정)
    target_lin_vel = 1.0
    lin_vel_error = vel_loc[:, 0] - target_lin_vel
    lin_vel_reward = torch.exp(-lin_vel_error**2) * track_lin_vel_xy_exp_weight

    # 5. 회전 속도 보상 (목표 yaw rate = 0)
    target_ang_vel = 0.0
    ang_vel_error = angvel_loc[:, 2] - target_ang_vel
    ang_vel_reward = torch.exp(-ang_vel_error**2) * track_ang_vel_z_exp_weight

    # 6. Torque 패널티
    torque_penalty = torch.sum(actions**2, dim=-1) * dof_torques_l2_weight

    # 7. Joint 가속도 패널티
    acc_penalty = torch.sum(dof_vel**2, dim=-1) * dof_acc_l2_weight

    # 8. Energy penalty
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # 9. Feet air time 보상 (feet_contact 센서 없으므로 0 처리)
    feet_air_time_reward = torch.zeros_like(progress_reward) * feet_air_time_weight

    # 10. 모든 보상 합산
    reward_terms = {
        "progress_reward": progress_reward,
        "alive_reward": alive_reward,
        "heading_reward": heading_reward,
        "orientation_penalty": orientation_penalty,
        "lin_vel_reward": lin_vel_reward,
        "ang_vel_reward": ang_vel_reward,
        "torque_penalty": torque_penalty,
        "acc_penalty": acc_penalty,
        "action_penalty": -actions_cost_scale * torch.sum(actions**2, dim=-1),
        "energy_penalty": -energy_cost_scale * electricity_cost,
        "feet_air_time_reward": feet_air_time_reward,
    }

    total_reward = torch.stack(list(reward_terms.values()), dim=0).sum(dim=0)

    # 11. 종료 시 death cost 적용
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)

    return total_reward, reward_terms


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
