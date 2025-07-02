# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Project CH."""

from isaaclab.utils import configclass

# Import master configuration
try:
    from go2_piper_master.tasks.direct.go2_piper_master.go2_piper_master_env_cfg import (
        Go2PiperMasterEnvCfg,
    )
except ImportError:
    # Fallback if master project is not installed
    print("Warning: go2_piper_master not found. Using minimal fallback config.")
    from isaaclab.envs import DirectRLEnvCfg
    Go2PiperMasterEnvCfg = DirectRLEnvCfg


@configclass
class ProjectChEnvCfg(Go2PiperMasterEnvCfg):
    """Environment configuration for Project CH.
    
    Inherits from master configuration and can be customized.
    """
    
    # Customize here as needed
    pass