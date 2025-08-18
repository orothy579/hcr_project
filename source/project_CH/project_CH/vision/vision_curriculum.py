import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs import ManagerBasedRLEnv
from typing import Sequence


@torch.no_grad()
def terrain_levels_vision(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    conf_thresh: float = 0.0,  # 0.0이면 신뢰도 무시(그냥 비교)
    up_margin: float = 0.0,  # 0.0이면 즉각 승급
    down_margin: float = 0.0,  # 0.0이면 즉각 강등
) -> torch.Tensor:
    """
    Vision 예측값으로 단순 승급/강등:
      - pred >= cur + up_margin     → 승급
      - pred <= cur - down_margin   → 강등
      - conf < conf_thresh          → 유지
    """
    # 타입 힌팅용 추출
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # 현재 레벨
    cur = terrain.terrain_levels[env_ids].to(env.device).float()

    # Vision 캐시 없으면 유지
    if not hasattr(env, "_vision_pred") or env._vision_pred is None:
        return torch.mean(terrain.terrain_levels.float())

    pred = env._vision_pred[env_ids].to(env.device).float()
    # conf 없으면 전부 True
    if (
        hasattr(env, "_vision_conf")
        and env._vision_conf is not None
        and conf_thresh > 0.0
    ):
        conf = env._vision_conf[env_ids].to(env.device).float()
        gate = conf >= conf_thresh
    else:
        gate = torch.ones_like(pred, dtype=torch.bool, device=env.device)

    # 승급/강등 마스크
    move_up = gate & (pred >= cur + up_margin)
    move_down = gate & (pred <= cur - down_margin) & (~move_up)

    # 레벨 갱신
    terrain.update_env_origins(env_ids, move_up, move_down)

    # 로깅(선택)
    env.extras["curr/vision_promote_frac"] = float(move_up.float().mean().item())
    env.extras["curr/vision_demote_frac"] = float(move_down.float().mean().item())

    return torch.mean(terrain.terrain_levels.float())
