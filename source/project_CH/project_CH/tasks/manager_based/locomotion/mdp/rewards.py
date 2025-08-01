# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contacts of undesired body parts (knees, thighs)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # shape: (num_envs, history_len, num_bodies)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
    
    # history와 body 차원을 모두 합산하여 (num_envs,) shape으로 만듦
    penalty = torch.sum(contact_forces, dim=(1, 2))
    return penalty


def desired_contacts(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Give reward when specific body parts are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 접촉 힘 계산
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_norm = contact_forces.norm(dim=-1).max(dim=1)[0]
    # 1N 이상 접촉 시 보상
    contacts = (contact_norm > 1.0).float()
    # body 개수 기준 합산
    reward = torch.sum(contacts, dim=1)
    return reward


import torch
import math


def get_leg_phase(env):
    """각 다리의 phase oscillator를 업데이트하여 반환.

    ORCAgym과 Zhang et al.(2024), Shao et al.(2022)의 연구 아이디어를 참고함.
    ground reaction force(GRF) 피드백을 통해 각 다리의 위상을 동기화.

    Returns:
        torch.Tensor: 업데이트된 각 다리의 phase 값.
    """
    # device 파악 (IsaacLab은 대부분 env.device 또는 env.sim_device 존재)
    device = getattr(env, "device", getattr(env, "sim_device", "cpu"))
    num_envs = getattr(env, "num_envs", 1)

    # phases 동적 생성 또는 device mismatch 시 재할당
    if (not hasattr(env, 'phases')) or (getattr(env, 'phases').device != device):
        env.phases = torch.rand(num_envs, 4, device=device) * 2 * torch.pi

    # obs, grf 없으면 더미 반환 (초기 dimension 체크용)
    if not hasattr(env, 'obs') or 'grf' not in getattr(env, 'obs', {}):
        return torch.zeros_like(env.phases, device=device)

    ph = env.phases
    grf = env.obs["grf"]
    if grf.device != device:
        grf = grf.to(device)
    cmd_vx = env.obs["base_vel"][:, 0].to(device)
    omega = torch.where(cmd_vx.abs() <= 0.5, 1.0, (1.5 + cmd_vx.abs()).clamp(max=4.0))
    sigma = torch.where(cmd_vx.abs() <= 0.5, 4.0, 1.0)
    xi = torch.where(cmd_vx.abs() <= 0.5, 1.0, 0.0)
    dphi = 2 * math.pi * omega - sigma * grf * (torch.cos(ph) + xi)
    ph = (ph + dphi * getattr(env, 'dt', 0.02)) % (2 * math.pi)
    env.phases = ph
    return ph


def phase_gait_reward(env):
    """phase와 GRF를 이용한 보상 함수.

    ORCAgym의 gait similarity 보상 구조를 참고하여 구현.
    stance/swing 타이밍이 위상과 일치할수록 보상을 높임.

    Returns:
        torch.Tensor: 각 환경 배치에 대한 보상 값.
    """
    device = getattr(env, "device", getattr(env, "sim_device", "cpu"))

    # obs가 아직 없으면 0 텐서 반환 (환경 초기화 단계 등)
    if not hasattr(env, "obs") or ("leg_phase" not in getattr(env, "obs", {})) or ("grf" not in getattr(env, "obs", {})):
        num_envs = getattr(env, "num_envs", 1)
        return torch.zeros(num_envs, device=device)

    ph = env.obs["leg_phase"]
    if ph.device != device:
        ph = ph.to(device)
    grf = env.obs["grf"]
    if grf.device != device:
        grf = grf.to(device)
    return -(grf * torch.sin(ph)).sum(dim=1)


def body_height_reward(env, target_height: float):
    # ManagerBasedRLEnv → InteractiveScene
    scene = env.scene

    # 로봇 객체 가져오기
    robot = scene["robot"]  # 딕셔너리처럼 접근

    # 월드 좌표계 기준 루트 z 위치 ==> sim2real 시 z 위치 알 수 있는 센서 필요
    z_pos = robot.data.root_pos_w[:, 2]

    return torch.exp(- (z_pos - target_height)**2 * 20.0)

