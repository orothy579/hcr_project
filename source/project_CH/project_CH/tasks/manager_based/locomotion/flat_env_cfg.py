from isaaclab.utils import configclass

# rough_env_cfg를 기반으로 flat 버전 구성
from .rough_env_cfg import Go2PiperRoughEnvCfg

@configclass
class Go2PiperFlatEnvCfg(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 보상 weight 변경 (flat 환경용)
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # Terrain을 평평한 plane으로 변경
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Height scanner 비활성화
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Terrain curriculum 제거
        self.curriculum.terrain_levels = None


@configclass
class Go2PiperFlatEnvCfg_PLAY(Go2PiperFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Play 모드 전용 설정
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Randomization 제거
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
