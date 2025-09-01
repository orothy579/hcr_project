from dataclasses import replace
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationTermCfg as ObsTerm,
    ObservationGroupCfg as ObsGroup,
    RewardTermCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from project_CH.tasks.manager_based.locomotion.mdp.rewards import (
    rew_action_arm_l2,
    rew_action_gripper_l2,
)


import isaacsim.core.utils.prims as prim_utils

from .rough_env_cfg import Go2PiperRoughEnvCfg


@configclass
class Go2PiperWholebodyEnvCfg(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.ee_cam = None

        # --- terrain/spacing 등 기존 설정 ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.scene.env_spacing = 5.0

        # --- RigidObjectCfg + sim_utils.CuboidCfg ---
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

        # --- Stage-1 locomotion pretrain  ---
        # (부모 cfg에 없는 필드여도 configclass에서는 attribute 추가 허용됨)
        self.action_schema = {
            "base_cmd": {"start": 0, "dim": 3},
            "arm_delta": {"start": 3, "dim": 4},
            "gripper": {"start": 7, "dim": 1},
        }
        self.stage_mode = "pretrain"

        # 팔/그리퍼 액션 L2 패널티 (커스텀 리워드 함수 이름은 프로젝트에 등록되어 있어야 함)
        self.rewards.arm_action_l2 = RewardTermCfg(
            func=rew_action_arm_l2,
            weight=-0.001,
            params={"slice_name": "arm_delta"},
        )
        self.rewards.gripper_action_l2 = RewardTermCfg(
            func=rew_action_gripper_l2,
            weight=-0.005,
            params={"slice_name": "gripper"},
        )
