# wbc를 위한

import torch
from isaaclab.utils.math import quat_apply, quat_inv
from isaaclab.managers import SceneEntityCfg


@torch.no_grad()
def get_object_pose_b(env, object: "SceneEntityCfg"):
    """물체(object)의 위치/자세를 로봇 base 좌표계로 변환."""
    pos_w, rot_w = env.scene.get_entity_pose(object.name)  # (num_envs,3), (num_envs,4)
    base_pos_w = env.robot.data.root_pos_w
    base_rot_w = env.robot.data.root_quat_w
    # world→base 변환
    pos_b = quat_apply(quat_inv(base_rot_w), pos_w - base_pos_w)
    return pos_b  # (num_envs,3)


@torch.no_grad()
def get_dest_pose_b(env, dest: "SceneEntityCfg"):
    """목표 구역(dest)의 중심 좌표를 base 좌표계로 변환."""
    pos_w, _ = env.scene.get_entity_pose(dest.name)
    base_pos_w = env.robot.data.root_pos_w
    base_rot_w = env.robot.data.root_quat_w
    pos_b = quat_apply(quat_inv(base_rot_w), pos_w - base_pos_w)
    return pos_b  # (num_envs,3)


@torch.no_grad()
def get_ee_pose_b(env, ee: "SceneEntityCfg"):
    """End-effector(piper_gripper_ee)의 위치를 base 좌표계로 변환."""
    pos_w, _ = env.scene.get_entity_pose(ee.name)
    base_pos_w = env.robot.data.root_pos_w
    base_rot_w = env.robot.data.root_quat_w
    pos_b = quat_apply(quat_inv(base_rot_w), pos_w - base_pos_w)
    return pos_b


@torch.no_grad()
def get_gripper_opening(env):
    """그리퍼 opening 값을 관측치로."""
    # 보통 조인트 두 개(piper_joint7,8) 사이 거리 or 각도를 쓰면 됨
    jpos = env.robot.data.joint_pos
    left = jpos[:, env.robot.joints["piper_joint7"].dof_idx]
    right = jpos[:, env.robot.joints["piper_joint8"].dof_idx]
    opening = left - right
    return opening.unsqueeze(-1)  # (num_envs,1)


@torch.no_grad()
def obs_zero_pad(env, dim: int) -> torch.Tensor:
    """(num_envs, dim)의 0 텐서 반환. Stage-1에서 policy obs 차원 고정을 위한 더미."""
    return torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float32)
