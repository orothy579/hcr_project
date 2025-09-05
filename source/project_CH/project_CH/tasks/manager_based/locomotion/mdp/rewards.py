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
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
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
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
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
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (
        torch.norm(command[:, :2], dim=1) < command_threshold
    )


def undesired_contacts(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize contacts of undesired body parts (knees, thighs)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # shape: (num_envs, history_len, num_bodies)
    contact_forces = contact_sensor.data.net_forces_w_history[
        :, :, sensor_cfg.body_ids, :
    ].norm(dim=-1)

    # history와 body 차원을 모두 합산하여 (num_envs,) shape으로 만듦
    penalty = torch.sum(contact_forces, dim=(1, 2))
    return penalty


def desired_contacts(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Give reward when specific body parts are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 접촉 힘 계산
    contact_forces = contact_sensor.data.net_forces_w_history[
        :, :, sensor_cfg.body_ids, :
    ]
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

    return torch.exp(-((z_pos - target_height) ** 2) * 20.0)


def suppress_leg_cross(env, vel_threshold: float = 0.3):
    """
    x축 명령 크기가 임계 이상일 때,
    - 전진(vx > +thr): 앞다리(FL, FR) 교차 패널티
    - 후진(vx < -thr): 뒷다리(RL, RR) 교차 패널티 => 필요시 추가 예정
    (월드 기준 y 간격 사용 — 기존 함수와 동일한 좌표계)
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command("base_velocity")  # (num_envs, 3)
    lin_x_cmd = cmd[:, 0]  # x축 명령어

    # --- 바디 이름 → 인덱스 매핑 가져오기 ---
    body_names = robot.data.body_names  # list of strings
    FL_idx = body_names.index("FL_foot")
    FR_idx = body_names.index("FR_foot")
    # HL_idx = body_names.index("HL_foot")
    # HR_idx = body_names.index("HR_foot")

    # --- 월드 기준 y 좌표 ---
    pos_y_FL = robot.data.body_pos_w[:, FL_idx, 1]
    pos_y_FR = robot.data.body_pos_w[:, FR_idx, 1]
    # pos_y_HL = robod.data.body_pos_w[]

    y_diff = torch.abs(pos_y_FL - pos_y_FR)
    crossing_penalty = torch.exp(-y_diff * 10.0)

    apply_mask = lin_x_cmd > vel_threshold
    penalty = torch.where(
        apply_mask, crossing_penalty, torch.zeros_like(crossing_penalty)
    )

    return penalty


def body_relative_height(
    env, sensor_cfg, target_clearance: float = 0.42, dbg: bool = False
):
    robot = env.scene["robot"]
    base_z = robot.data.root_pos_w[:, 2]  # 루트 z (월드)

    hs = env.scene.sensors[sensor_cfg.name]  # height_scanner
    hits_w = hs.data.ray_hits_w  # [N_env, N_rays, 3]
    z = hits_w[..., 2]  # 충돌 지점 z
    valid = torch.isfinite(z)  # inf(미충돌) 마스크

    # 유효 레이만으로 지면 z를 보수적으로 추정(상위 분위수/최대값 등)
    z_masked = torch.where(valid, z, z.new_full(z.shape, -1e6))
    # 계단 모서리 등을 고려해 90% 분위수 사용(더 보수적으로는 max)
    terrain_z = torch.quantile(z_masked, q=0.9, dim=1)

    clearance = base_z - terrain_z
    return torch.exp(-((clearance - target_clearance) ** 2) * 20.0)


def _find_global_idx(all_names, target):
    try:
        return all_names.index(target)
    except ValueError:
        return None


def contact_balance(env, sensor_cfg):
    cs = env.scene.sensors[sensor_cfg.name]
    # 전체 바디에 대한 접촉 이력 → (N, B)
    contact_norm = cs.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]
    contacts_all = contact_norm > 1.0

    names_all = env.scene["robot"].data.body_names
    gFL = _find_global_idx(names_all, "FL_foot")
    gFR = _find_global_idx(names_all, "FR_foot")
    gHL = _find_global_idx(names_all, "HL_foot")
    gHR = _find_global_idx(names_all, "HR_foot")

    missing = [
        s for s, g in [("FL", gFL), ("FR", gFR), ("HL", gHL), ("HR", gHR)] if g is None
    ]
    if missing:
        raise RuntimeError(
            f"[contact_balance] feet {missing} not found in body names: {names_all}"
        )

    left = contacts_all[:, [gFL, gHL]].sum(dim=1)
    right = contacts_all[:, [gFR, gHR]].sum(dim=1)
    diff = torch.abs(left - right)
    return torch.exp(-diff)


def side_support(env, sensor_cfg):
    cs = env.scene.sensors[sensor_cfg.name]
    contact_norm = cs.data.net_forces_w_history.norm(dim=-1).max(dim=1)[0]
    contacts_all = contact_norm > 1.0

    names_all = env.scene["robot"].data.body_names
    gFL = _find_global_idx(names_all, "FL_foot")
    gFR = _find_global_idx(names_all, "FR_foot")
    gHL = _find_global_idx(names_all, "HL_foot")
    gHR = _find_global_idx(names_all, "HR_foot")

    missing = [
        s for s, g in [("FL", gFL), ("FR", gFR), ("HL", gHL), ("HR", gHR)] if g is None
    ]
    if missing:
        raise RuntimeError(
            f"[side_support] feet {missing} not found in body names: {names_all}"
        )

    left_has = contacts_all[:, [gFL, gHL]].any(dim=1).float()
    right_has = contacts_all[:, [gFR, gHR]].any(dim=1).float()
    return 0.5 + 0.5 * (left_has * right_has)


# ---- wbc 관련 보상 ---
from isaaclab.utils.math import quat_apply, quat_inv


# ---------- 공통 유틸 ----------
def _asset_root_pose_w(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.root_pos_w, asset.data.root_quat_w  # (N,3),(N,4)


def _body_pose_w(env, asset_name: str, body_name: str):
    asset = env.scene[asset_name]
    idx = asset.data.body_names.index(body_name)
    return asset.data.body_pos_w[:, idx, :], asset.data.body_quat_w[:, idx, :]


def _to_base(pos_w, base_pos_w, base_quat_w):
    return quat_apply(quat_inv(base_quat_w), pos_w - base_pos_w)


def get_gripper_opening_generic(env) -> torch.Tensor:
    """
    그리퍼 opening을 조인트 또는 링크 사이 거리로 추정.
    - 조인트명이 없는 세팅을 대비해 링크 거리 fallback 포함.
    반환: (N,1)
    """
    robot = env.scene["robot"]
    names = robot.data.body_names
    i7 = names.index("piper_link7")
    i8 = names.index("piper_link8")
    pos = robot.data.body_pos_w
    return torch.norm(pos[:, i7, :] - pos[:, i8, :], dim=-1, keepdim=True)


# --- 스텝 마스크 util ---
def _soft_gate_ge(x, thr, k=40.0):
    # x >= thr 에서 1, 작아질수록 0으로 부드럽게 (sigmoid)
    return torch.sigmoid(k * (x - thr))


def _soft_gate_le(x, thr, k=40.0):
    # x <= thr 에서 1
    return torch.sigmoid(k * (thr - x))


def _inside_zone_xy(obj_pos_w, zone_pos_w, half_xy):
    dx = (obj_pos_w[:, 0] - zone_pos_w[:, 0]).abs()
    dy = (obj_pos_w[:, 1] - zone_pos_w[:, 1]).abs()
    return (dx <= half_xy[0]) & (dy <= half_xy[1])


def _phase_masks(
    env,
    dist_align=0.90,  # NAV→ALIGN 경계
    dist_grasp=0.10,  # ALIGN→GRASP 경계
    close_open=0.02,  # '닫힘' 임계 (opening)
    zone_half_xy=(0.25, 0.25),
):
    # EE / Object / Zone
    obj_pos_w, _ = _asset_root_pose_w(env, "object_box")
    ee_pos_w, _ = _body_pose_w(env, "robot", "piper_gripper_base")
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)  # (N,)
    opening = get_gripper_opening_generic(env).squeeze(-1).abs()

    zone_pos_w, _ = _asset_root_pose_w(env, "place_zone")
    inside_zone = _inside_zone_xy(obj_pos_w, zone_pos_w, zone_half_xy)

    # carrying(잡음) 판단을 부드럽게: close & 근접
    closed_mask = _soft_gate_le(opening, close_open)  # 닫힘일수록 1
    near_for_grasp = _soft_gate_le(dist, dist_grasp)  # 10cm 내면 1
    carrying_soft = closed_mask * near_for_grasp  # [0..1]

    # 단계별 소프트 마스크 (값은 0~1)
    m_nav = _soft_gate_ge(dist, dist_align) * (
        1.0 - carrying_soft
    )  # 멀리 + 아직 안 집음
    m_align = (
        (1.0 - _soft_gate_ge(dist, dist_align))
        * (1.0 - near_for_grasp)
        * (1.0 - carrying_soft)
    )
    m_grasp = near_for_grasp * (1.0 - carrying_soft)  # 근접인데 아직 미집음
    m_carry = carrying_soft * (1.0 - inside_zone.float())  # 집은 상태 + Zone 밖
    m_place = carrying_soft * inside_zone.float()  # 집은 상태 + Zone 안
    return m_nav, m_align, m_grasp, m_carry, m_place


def rew_nav_to_object(
    env, dist_scale: float = 0.8, fwd_gain: float = 0.1, yaw_gain: float = 0.4
):
    # poses
    base_pos_w = env.scene["robot"].data.root_pos_w
    base_quat_w = env.scene["robot"].data.root_quat_w
    obj_pos_w, _ = _asset_root_pose_w(env, "object_box")
    # 방향/거리
    d_w = obj_pos_w - base_pos_w  # (N,3)
    d_xy = d_w[:, :2]
    dist = torch.norm(d_xy, dim=-1) + 1e-6
    dir_xy = d_xy / dist.unsqueeze(-1)  # 단위벡터
    # 진행 속도: base 선속도를 yaw frame으로 정렬
    vel_yaw = quat_apply_inverse(
        yaw_quat(base_quat_w), env.scene["robot"].data.root_lin_vel_w[:, :3]
    )[:, :2]
    speed_along = (vel_yaw * dir_xy).sum(dim=-1).clamp(min=0.0)  # 목표방향 성분만
    # 거리 shaping + 진행 보상 + 요 정렬(전방 x축과 dir_xy 코사인)
    approach = torch.exp(-((dist / dist_scale) ** 2))
    base_x = torch.tensor([1.0, 0.0], device=env.device).expand_as(dir_xy)
    cos_yaw = (dir_xy * base_x).sum(dim=-1).clamp(-1, 1)
    r = approach + fwd_gain * speed_along + yaw_gain * cos_yaw
    m_nav, m_align, _, _, _ = _phase_masks(env)
    # NAV + ALIGN에서만 유효 (너가 원하는대로 더 좁혀도 됨)
    return r * torch.clamp(m_nav + m_align, 0.0, 1.0)


def rew_nav_to_zone(
    env, dist_scale: float = 0.8, fwd_gain: float = 0.1, yaw_gain: float = 0.4
):
    base_pos_w = env.scene["robot"].data.root_pos_w
    base_quat_w = env.scene["robot"].data.root_quat_w
    zone_pos_w, _ = _asset_root_pose_w(env, "place_zone")
    d_w = zone_pos_w - base_pos_w
    d_xy = d_w[:, :2]
    dist = torch.norm(d_xy, dim=-1) + 1e-6
    dir_xy = d_xy / dist.unsqueeze(-1)
    vel_yaw = quat_apply_inverse(
        yaw_quat(base_quat_w), env.scene["robot"].data.root_lin_vel_w[:, :3]
    )[:, :2]
    speed_along = (vel_yaw * dir_xy).sum(dim=-1).clamp(min=0.0)
    approach = torch.exp(-((dist / dist_scale) ** 2))
    base_x = torch.tensor([1.0, 0.0], device=env.device).expand_as(dir_xy)
    cos_yaw = (dir_xy * base_x).sum(dim=-1).clamp(-1, 1)
    r = approach + fwd_gain * speed_along + yaw_gain * cos_yaw
    _, _, _, m_carry, _ = _phase_masks(env)
    # 보통 Zone으로 향하는 건 carry 중 유효
    return r * m_carry


# ---------- 0) 접근(EE→Object) ----------
def rew_approach_ee_object(
    env,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    dist_scale: float = 0.06,
    use_base_frame: bool = True,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    ee_body = (
        ee_cfg.body_names[0]
        if getattr(ee_cfg, "body_names", None)
        else "piper_gripper_base"
    )
    ee_pos_w, _ = _body_pose_w(env, ee_cfg.name, ee_body)
    if use_base_frame:
        base_pos_w = env.scene["robot"].data.root_pos_w
        base_quat_w = env.scene["robot"].data.root_quat_w
        obj_pos = _to_base(obj_pos_w, base_pos_w, base_quat_w)
        ee_pos = _to_base(ee_pos_w, base_pos_w, base_quat_w)
    else:
        obj_pos, ee_pos = obj_pos_w, ee_pos_w
    dist = torch.norm(obj_pos - ee_pos, dim=-1)

    out = torch.exp(-((dist / dist_scale) ** 2))
    _, m_align, m_grasp, _, _ = _phase_masks(env)
    return out * torch.clamp(m_align + m_grasp, 0.0, 1.0)


# ---------- 1) 조기닫힘 페널티(멀리서 닫으면 -) ----------
def pen_premature_close(
    env,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    close_thresh: float = 0.02,  # opening 이 값보다 작으면 '닫힘'으로 간주
    far_dist: float = 0.15,  # EE-Object 거리가 이보다 크면 '멀리'
    use_base_frame: bool = True,
    weight: float = 1.0,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    ee_body = (
        ee_cfg.body_names[0]
        if getattr(ee_cfg, "body_names", None)
        else "piper_gripper_base"
    )
    ee_pos_w, _ = _body_pose_w(env, ee_cfg.name, ee_body)
    if use_base_frame:
        base_pos_w = env.scene["robot"].data.root_pos_w
        base_quat_w = env.scene["robot"].data.root_quat_w
        obj_pos = _to_base(obj_pos_w, base_pos_w, base_quat_w)
        ee_pos = _to_base(ee_pos_w, base_pos_w, base_quat_w)
    else:
        obj_pos, ee_pos = obj_pos_w, ee_pos_w
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    opening = get_gripper_opening_generic(env).squeeze(-1).abs()
    closed_far = (opening < close_thresh) & (dist > far_dist)

    base = -weight * closed_far.float()
    m_nav, m_align, _, _, _ = _phase_masks(env)
    return base * torch.clamp(m_nav + m_align, 0.0, 1.0)


# ---------- 2) 그랩(근접 + 닫힘) ----------
def rew_grasp_soft(
    env,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    ee_obj_dist_scale: float = 0.06,
    grip_close_scale: float = 0.01,
    use_base_frame: bool = True,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    ee_body = (
        ee_cfg.body_names[0]
        if getattr(ee_cfg, "body_names", None)
        else "piper_gripper_base"
    )
    ee_pos_w, _ = _body_pose_w(env, ee_cfg.name, ee_body)
    if use_base_frame:
        base_pos_w = env.scene["robot"].data.root_pos_w
        base_quat_w = env.scene["robot"].data.root_quat_w
        obj_pos = _to_base(obj_pos_w, base_pos_w, base_quat_w)
        ee_pos = _to_base(ee_pos_w, base_pos_w, base_quat_w)
    else:
        obj_pos, ee_pos = obj_pos_w, ee_pos_w
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    opening = get_gripper_opening_generic(env).squeeze(-1).abs()
    prox_term = torch.exp(-((dist / ee_obj_dist_scale) ** 2))
    grip_term = torch.exp(-((opening / grip_close_scale) ** 2))
    _, _, m_grasp, _, _ = _phase_masks(env)
    return (prox_term * grip_term) * m_grasp


# ---------- 3) 운반 안정성(EE 근처 유지 + 흔들림 억제) ----------
def rew_carry_stability(
    env,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    dist_scale: float = 0.06,
    ang_vel_penalty: float = 0.05,  # base 각속도 패널티 비중
    use_base_frame: bool = True,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    ee_body = (
        ee_cfg.body_names[0]
        if getattr(ee_cfg, "body_names", None)
        else "piper_gripper_base"
    )
    ee_pos_w, _ = _body_pose_w(env, ee_cfg.name, ee_body)
    if use_base_frame:
        base_pos_w = env.scene["robot"].data.root_pos_w
        base_quat_w = env.scene["robot"].data.root_quat_w
        obj_pos = _to_base(obj_pos_w, base_pos_w, base_quat_w)
        ee_pos = _to_base(ee_pos_w, base_pos_w, base_quat_w)
    else:
        obj_pos, ee_pos = obj_pos_w, ee_pos_w
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    prox = torch.exp(-((dist / dist_scale) ** 2))
    yaw_rate = env.scene["robot"].data.root_ang_vel_w[:, 2].abs()
    stabl = torch.exp(-ang_vel_penalty * yaw_rate)
    _, _, _, m_carry, _ = _phase_masks(env)
    return (prox * stabl) * m_carry


# ---------- 4) 드랍 페널티 ----------
def pen_drop(
    env,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    open_thresh: float = 0.03,  # opening 커지면 놓은 것으로 간주
    far_dist: float = 0.20,  # EE-Object 거리가 커지면 놓은 것으로 간주
    z_fall_delta: float = 0.06,  # z 급하강 허용치
    use_base_frame: bool = False,
    weight: float = 3.0,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    ee_body = (
        ee_cfg.body_names[0]
        if getattr(ee_cfg, "body_names", None)
        else "piper_gripper_base"
    )
    ee_pos_w, _ = _body_pose_w(env, ee_cfg.name, ee_body)
    if use_base_frame:
        base_pos_w = env.scene["robot"].data.root_pos_w
        base_quat_w = env.scene["robot"].data.root_quat_w
        obj_pos = _to_base(obj_pos_w, base_pos_w, base_quat_w)
        ee_pos = _to_base(ee_pos_w, base_pos_w, base_quat_w)
    else:
        obj_pos, ee_pos = obj_pos_w, ee_pos_w

    opening = get_gripper_opening_generic(env).squeeze(-1).abs()
    dist = torch.norm(obj_pos - ee_pos, dim=-1)

    # z 급하강: 직전 step의 obj z와 비교
    if not hasattr(env, "_prev_obj_z"):
        env._prev_obj_z = obj_pos[:, 2].detach()
        z_fall = torch.zeros_like(dist)
    else:
        z_fall = (env._prev_obj_z - obj_pos[:, 2]) > z_fall_delta
        env._prev_obj_z = obj_pos[:, 2].detach()

    # --- 여기 수정 ---
    cond1 = opening > open_thresh
    cond2 = dist > far_dist
    cond3 = z_fall.bool()  # 이미 bool이면 그대로, float이면 강제 변환
    dropped = cond1 | cond2 | cond3

    _, _, _, m_carry, m_place = _phase_masks(env)
    return (-weight * dropped.float()) * torch.clamp(m_carry + m_place, 0.0, 1.0)


# ---------- 5) 플레이스(Zone xy 근접 + 높이 OK) ----------
def rew_place_soft(
    env,
    object_cfg: SceneEntityCfg,
    zone_cfg: SceneEntityCfg,
    zone_half_size_xy: tuple[float, float] = (0.25, 0.25),
    height_tolerance: float = 0.08,
    use_base_frame: bool = False,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    zone_pos_w, _ = _asset_root_pose_w(env, zone_cfg.name)
    if use_base_frame:
        base_pos_w = env.scene["robot"].data.root_pos_w
        base_quat_w = env.scene["robot"].data.root_quat_w
        obj_pos = _to_base(obj_pos_w, base_pos_w, base_quat_w)
        zone_pos = _to_base(zone_pos_w, base_pos_w, base_quat_w)
    else:
        obj_pos, zone_pos = obj_pos_w, zone_pos_w
    dx = (obj_pos[:, 0] - zone_pos[:, 0]) / zone_half_size_xy[0]
    dy = (obj_pos[:, 1] - zone_pos[:, 1]) / zone_half_size_xy[1]
    xy_term = torch.exp(-(dx**2 + dy**2))
    z_ok = (obj_pos[:, 2] >= (zone_pos[:, 2] - height_tolerance)).float()

    out = xy_term * (0.5 + 0.5 * z_ok)
    _, _, _, _, m_place = _phase_masks(env)
    return out * m_place


# ---------- 6) 릴리즈 보너스(Zone 안에서 벌리면 +, 밖이면 -) ----------
def rew_place_release(
    env,
    object_cfg: SceneEntityCfg,
    zone_cfg: SceneEntityCfg,
    zone_half_size_xy: tuple[float, float] = (0.25, 0.25),
    open_increase: float = 0.01,  # opening 증가량
    weight_out_pen: float = 1.0,
) -> torch.Tensor:
    obj_pos_w, _ = _asset_root_pose_w(env, object_cfg.name)
    zone_pos_w, _ = _asset_root_pose_w(env, zone_cfg.name)
    dx = (obj_pos_w[:, 0] - zone_pos_w[:, 0]).abs() <= zone_half_size_xy[0]
    dy = (obj_pos_w[:, 1] - zone_pos_w[:, 1]).abs() <= zone_half_size_xy[1]
    inside = dx & dy

    opening = get_gripper_opening_generic(env).squeeze(-1)
    if not hasattr(env, "_prev_opening"):
        env._prev_opening = opening.detach()
        delta_open = torch.zeros_like(opening)
    else:
        delta_open = opening - env._prev_opening
        env._prev_opening = opening.detach()

    pos = (inside & (delta_open > open_increase)).float()
    neg = ((~inside) & (delta_open > open_increase)).float()
    out = pos - weight_out_pen * neg
    _, _, _, _, m_place = _phase_masks(env)
    return out * m_place


# ---------- 7) body height 관련 보상 ----------
def body_relative_height_gated(
    env,
    sensor_cfg,
    target_clearance: float = 0.42,
    dist_align: float = 0.30,
    dist_grasp: float = 0.10,
    close_open: float = 0.02,
):
    # 원래 body_relative_height 계산 그대로
    base = body_relative_height(env, sensor_cfg, target_clearance=target_clearance)
    m_nav, m_align, m_grasp, m_carry, m_place = _phase_masks(
        env, dist_align=dist_align, dist_grasp=dist_grasp, close_open=close_open
    )
    # NAV/CARRY에서만 유효(= 그 외 단계에서는 0)
    gate = torch.clamp(m_nav + m_carry, 0.0, 1.0)
    return base * gate
