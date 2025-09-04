# wbc를 위한 observation

import torch
from isaaclab.utils.math import quat_apply, quat_inv
from isaaclab.managers import SceneEntityCfg


# helpers
def _asset_root_pose_w(env, asset_name: str):
    asset = env.scene[asset_name]
    return asset.data.root_pos_w, asset.data.root_quat_w  # (N,3),(N,4)


def _body_pose_w(env, asset_name: str, body_name: str):
    asset = env.scene[asset_name]
    idx = asset.data.body_names.index(body_name)
    return asset.data.body_pos_w[:, idx, :], asset.data.body_quat_w[:, idx, :]


@torch.no_grad()
def get_object_pose_b(env, object: "SceneEntityCfg"):
    pos_w, _ = _asset_root_pose_w(env, object.name)
    base_pos_w = env.scene["robot"].data.root_pos_w
    base_rot_w = env.scene["robot"].data.root_quat_w
    return quat_apply(quat_inv(base_rot_w), pos_w - base_pos_w)  # (N,3)


@torch.no_grad()
def get_dest_pose_b(env, dest: "SceneEntityCfg"):
    pos_w, _ = _asset_root_pose_w(env, dest.name)
    base_pos_w = env.scene["robot"].data.root_pos_w
    base_rot_w = env.scene["robot"].data.root_quat_w
    return quat_apply(quat_inv(base_rot_w), pos_w - base_pos_w)  # (N,3)


@torch.no_grad()
def get_ee_pose_b(env, ee: "SceneEntityCfg"):
    body = ee.body_names[0] if getattr(ee, "body_names", None) else "piper_gripper_base"
    pos_w, _ = _body_pose_w(env, ee.name, body)  # ee.name == "robot"
    base_pos_w = env.scene["robot"].data.root_pos_w
    base_rot_w = env.scene["robot"].data.root_quat_w
    return quat_apply(quat_inv(base_rot_w), pos_w - base_pos_w)  # (N,3)


@torch.no_grad()
def get_gripper_opening(env):
    # 링크 거리 기반 (조인트 맵 의존 제거)
    robot = env.scene["robot"]
    names = robot.data.body_names
    i7 = names.index("piper_link7")
    i8 = names.index("piper_link8")
    pos = robot.data.body_pos_w
    d = torch.norm(pos[:, i7, :] - pos[:, i8, :], dim=-1, keepdim=True)
    return d  # (N,1)


@torch.no_grad()
def obs_zero_pad(env, dim: int) -> torch.Tensor:
    """(num_envs, dim)의 0 텐서 반환. Stage-1에서 policy obs 차원 고정을 위한 더미."""
    return torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float32)
