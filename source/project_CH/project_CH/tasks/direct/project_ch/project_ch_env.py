# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project CH environment - inherits from go2_piper_master."""

from __future__ import annotations

import torch

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


class ProjectChEnv(Go2PiperMasterEnv):
    """Project CH environment - extends master environment."""
    
    cfg: ProjectChEnvCfg

    def __init__(self, cfg: ProjectChEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize Project CH environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        if MASTER_AVAILABLE:
            print(f"[INFO] Project CH initialized with {cfg.scene.num_envs} environments")
        else:
            print("[WARNING] Running with fallback environment")

    def _get_rewards(self) -> torch.Tensor:
        """CH-specific reward function."""
        if not MASTER_AVAILABLE:
            return torch.zeros(self.num_envs, device=self.device)
            
        # Use master rewards as base - can be customized later
        return super()._get_rewards()

    def _get_observations(self) -> dict:
        """CH-specific observations.""" 
        if not MASTER_AVAILABLE:
            return {"policy": torch.zeros((self.num_envs, 20), device=self.device)}
            
        # Use master observations as base - can be customized later
        return super()._get_observations()