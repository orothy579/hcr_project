# Baseline(고정 난이도, 비전 없음)
from isaaclab.utils import configclass
from project_CH.tasks.manager_based.locomotion.mdp.rewards import (
    body_height_reward,
    suppress_leg_cross,
)
from isaaclab.managers import SceneEntityCfg, RewardTermCfg

# rough_env_cfg를 상속
from .rough_env_cfg import Go2PiperRoughEnvCfg


@configclass
class Go2PiperBaseEnvCfg(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Terrain을 평평한 plane으로 변경
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
