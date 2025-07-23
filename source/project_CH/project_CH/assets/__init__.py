# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Assets for Project CH - inherits from go2_piper_master."""

# Import shared robot configuration from master project
try:
    # First, try nested path (editable install layout)
    from go2_piper_master.go2_piper_master.assets import GO2_PIPER_CFG  # type: ignore
except ImportError:
    try:
        # Fallback: legacy path
        from go2_piper_master.assets import GO2_PIPER_CFG  # type: ignore
    except ImportError:
        print("[WARNING] go2_piper_master not found. Using DummyGo2Cfg as fallback.")
        from .dummy_go2_cfg import DummyGo2Cfg as GO2_PIPER_CFG

# export
__all__ = ["GO2_PIPER_CFG"] 