from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationTermCfg as ObsTerm,
    ObservationGroupCfg as ObsGroup,
    RewardTermCfg,
)

from .rough_env_cfg import Go2PiperRoughEnvCfg


@configclass
class Go2PiperWholebodyEnvCfg(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- WBC 관련 4개 관측 정보 추가 ---
        setattr(
            self.observations.policy,
            "wbc_obj_pose_b",
            ObsTerm(
                func="project_CH.tasks.manager_based.locomotion.mdp.observation:obs_zero_pad",
                params={"dim": 3},
            ),
        )
        setattr(
            self.observations.policy,
            "wbc_dest_pose_b",
            ObsTerm(
                func="project_CH.tasks.manager_based.locomotion.mdp.observation:obs_zero_pad",
                params={"dim": 3},
            ),
        )
        setattr(
            self.observations.policy,
            "wbc_ee_pose_b",
            ObsTerm(
                func="project_CH.tasks.manager_based.locomotion.mdp.observation:obs_zero_pad",
                params={"dim": 3},
            ),  # 필요 시 6으로
        )
        setattr(
            self.observations.policy,
            "wbc_grip_open",
            ObsTerm(
                func="project_CH.tasks.manager_based.locomotion.mdp.observation:obs_zero_pad",
                params={"dim": 1},
            ),
        )

        # concat 모드가 꺼져 있으면 켜주기 (term들을 하나의 벡터로 이어붙임)
        self.observations.policy.concatenate_terms = True

        # --- 물체 , place zone 생성 ---
        self.scene.object_box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cuboid",
            spawn=sim_utils.CuboidCfg(
                size=(0.10, 0.10, 0.10),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    rest_offset=0.0, contact_offset=0.005
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

        self.scene.place_zone = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PlaceZone",
            spawn=sim_utils.CuboidCfg(
                size=(0.50, 0.50, 0.02),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True, kinematic_enabled=True
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.8, 1.0)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

        self.scene.object_box.init_state.pos = (0.7, -0.25, 0.2)
        self.scene.place_zone.init_state.pos = (1.5, 0.45, 0.01)

        # --- Stage-1 locomotion pretrain  ---
        # (부모 cfg에 없는 필드여도 configclass에서는 attribute 추가 허용됨)
        self.action_schema = {
            "base_cmd": {"start": 0, "dim": 3},
            "arm_delta": {"start": 3, "dim": 4},
            "gripper": {"start": 7, "dim": 1},
        }
        self.stage_mode = "pretrain"

        self.scene.ee_cam = None

        # --- terrain/spacing 등 기존 설정 ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.scene.env_spacing = 5.0

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
