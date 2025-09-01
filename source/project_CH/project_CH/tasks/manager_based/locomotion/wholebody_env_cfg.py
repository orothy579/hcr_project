# Baseline(고정 난이도, 비전 없음)
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, ObservationTermCfg
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

# rough_env_cfg를 상속
from .rough_env_cfg import Go2PiperRoughEnvCfg


@configclass
class Go2PiperWholebodyEnvCfg(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.ee_cam = None

        # ======================== Terrain을 평평한 plane으로 변경 ========================
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # ======================== 로봇마다 거리 띄우기 ========================
        self.scene.env_spacing = 5

        # ======================== pick & place asset setting ========================
        prim_utils.create_prim("/World/Objects", "Xform")

        cfg_cuboid = sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
        cfg_cuboid.func(
            "/World/Objects/Cuboid1", cfg_cuboid, translation=(-1.0, 1.0, 1.0)
        )
