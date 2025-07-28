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
        self.action_scale = 1.0
        self.joint_gears = torch.ones(self.robot.num_joints, device=self.device)
        self.motor_effort_ratio = torch.one_like(self.joint_gears)
        
        self.potentials = torch.zeros(self.num_envs, device=self.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        
        self.targets = torch.tensor([1000,0,0], device=self.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        
        self.start_rotation = torch.tensor([1,0,0,0], device=self.device)
        self.inv_start_rot = quat_conjugate(se)

    def _get_rewards(self) -> torch.Tensor:
        """CH-specific reward function."""
        if not MASTER_AVAILABLE:
            return torch.zeros(self.num_envs, device=self.device)
        
        # master base alive reward
        base_reward = super()._get_rewards()
        
        locomotion_reward = self._compute_ch_specific_rewards()
            
        return base_reward + locomotion_reward

    def _get_observations(self) -> dict:
        """CH-specific observations.""" 
        if not MASTER_AVAILABLE:
            return {"policy": torch.zeros((self.num_envs, 20), device=self.device)}
            
        # Use master observations as base - can be customized later
        return super()._get_observations()
    