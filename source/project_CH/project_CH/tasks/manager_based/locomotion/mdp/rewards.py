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


import torch


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


def body_height_reward(env, target_height: float):
    # ManagerBasedRLEnv → InteractiveScene
    scene = env.scene

    # 로봇 객체 가져오기
    robot = scene["robot"]  # 딕셔너리처럼 접근

    # 월드 좌표계 기준 루트 z 위치 ==> sim2real 시 z 위치 알 수 있는 센서 필요
    z_pos = robot.data.root_pos_w[:, 2]

    return torch.exp(- (z_pos - target_height)**2 * 20.0)


def suppress_front_leg_crossing_when_forward(env, vel_threshold: float = 0.3):
    """
    x축 명령어가 일정 이상일 때 앞다리가 서로 교차하면 penalty.
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command("base_velocity")  # (num_envs, 3)
    lin_x_cmd = cmd[:, 0]  # x축 명령어

    # --- 바디 이름 → 인덱스 매핑 가져오기 ---
    body_names = robot.data.body_names  # list of strings
    FL_idx = body_names.index("FL_foot")
    FR_idx = body_names.index("FR_foot")

    # --- 월드 기준 y 좌표 ---
    pos_y_FL = robot.data.body_pos_w[:, FL_idx, 1]
    pos_y_FR = robot.data.body_pos_w[:, FR_idx, 1]

    y_diff = torch.abs(pos_y_FL - pos_y_FR)
    crossing_penalty = torch.exp(-y_diff * 10.0)

    apply_mask = lin_x_cmd > vel_threshold
    penalty = torch.where(apply_mask, crossing_penalty, torch.zeros_like(crossing_penalty))

    return penalty
