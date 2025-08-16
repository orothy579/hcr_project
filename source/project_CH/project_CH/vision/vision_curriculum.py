import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from typing import Sequence
from isaaclab.envs import ManagerBasedRLEnv

@torch.no_grad()
def terrain_levels_vision(
  env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  conf_thresh: float =0.6, allow_down_bias: int = 1
):
  """
  Vision(pred/conf)으로 승급/강등을 게이팅
  - conf < conf_thres : 변동 없음
  - pred >= 현재레벨 : 승급
  - pred + allow_down_bias < 현재레벨 : 강등

  Args:
      env (ManagerBasedRLEnv): _description_
      env_ids (Sequence[int]): _description_
      asset_cfg (SceneEntityCfg, optional): _description_. Defaults to SceneEntityCfg("robot").
      conf_thresh (float, optional): _description_. Defaults to 0.6.
      allow_down_bias (int, optional): _description_. Defaults to 1.
  """
  
  # 참조
  asset: Articulation = env.scene[asset_cfg.name]
  terrain: TerrainImporter = env.scene.terrain
  
  # 현재 레벨
  cur_levels = terrain.terrain_levels[env_ids].to(env.device)
  
  # vision cash
  if not hasattr(env, "_vision_pred") or env._vision_pred is None:
    return  torch.mean(terrain.terrain_levels.float())
  pred = env._vision_pred[env_ids].to(env.device)
  conf = env._vision_conf[env_ids].to(env.device)
  high_conf = conf >= conf_thresh
  
  # vision based mask
  allow_up = high_conf & (pred >= cur_levels)
  allow_down = high_conf & (pred + allow_down_bias < cur_levels)
  
  # 기존 거리/명령 기반 기준 추가 가능
  move_up = allow_up
  move_down = allow_down & ~move_up # 충돌 방지
  
  # level update
  terrain.update_env_origins(env_ids, move_up, move_down)
  
  # loggin (평균 승급/강등 비율)
  env.extras["curr/vision_promote_frac"] = float(move_up.float().mean().item())
  env.extras["curr/vision_demote_frac"] = float(move_down.float().mean().item())
  
  return torch.mean(terrain.terrain_levels.float())