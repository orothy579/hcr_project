# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configurations for Project CH."""

# Re-export PPO runner configurations for easy import by gym registry
from .rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
    UnitreeGo2FlatPPORunnerCfg,
)

__all__ = [
    "UnitreeGo2RoughPPORunnerCfg",
    "UnitreeGo2FlatPPORunnerCfg",
]