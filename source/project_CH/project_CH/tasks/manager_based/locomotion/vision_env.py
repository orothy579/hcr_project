import torch
import torch.nn.functional as F
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from project_CH.tasks.manager_based.locomotion.vision_env_cfg import Go2PiperVisionEnvCfg

# 이미지 전처리
def preprocess_rgb(env, cam_name="ee_cam", size=96):
  """EE cam 이미지를 (N,3,size, size) float32로 전처리

  Args:
      env (_type_): _description_
      cam_name (str, optional): _description_. Defaults to "ee_cam".
      size (int, optional): _description_. Defaults to 96.
  """
  x = env.scene.sensors[cam_name].data.rgb
  if x.dtype != torch.float32:
      x = x.float()
  if x.max() > 1.5:
      x = x / 255.0
  x = x.clamp(0., 1.).permute(0, 3, 1, 2).contiguous()
  if x.shape[-2:] != (size, size):
      x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
  return x

# env class
class Go2PiperVisionEnv(ManagerBasedRLEnv):
  def __init__(self, cfg: Go2PiperVisionEnvCfg, *args, **kwargs):
    super().__init__(cfg, *args, **kwargs)
    self._global_step = 0
    
  def _vision_smoke(self):
    x = preprocess_rgb(self, "ee_cam", 96)
    self.extras["vision/shape_ok"] = float(x.shape[-1] == 96)
    self.extras["vision/device_cuda"] = float(x.is_cuda)
    
  def post_physics_step(self):
    super().post_physics_step()
    self._global_step += 1
  
  def _reset_idx(self, env_ids: torch.Tensor):
    if env_ids.numel() > 0:
      self._vision_smoke()
    super()._reset_idx(env_ids)