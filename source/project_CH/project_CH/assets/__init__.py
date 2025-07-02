# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Assets for Project CH - inherits from go2_piper_master."""

# Import shared robot configuration from master project
try:
    from go2_piper_master.assets import GO2_PIPER_CFG
    __all__ = ["GO2_PIPER_CFG"]
except ImportError:
    print("[WARNING] go2_piper_master not found. Please install master project first.")
    __all__ = [] 