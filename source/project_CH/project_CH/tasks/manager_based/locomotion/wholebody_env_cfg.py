from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationTermCfg as ObsTerm,
    ObservationGroupCfg as ObsGroup,
    RewardTermCfg,
)

from project_CH.tasks.manager_based.locomotion.mdp.observation import (
    get_dest_pose_b,
    get_ee_pose_b,
    get_gripper_opening,
    get_object_pose_b,
    obs_zero_pad,
)

from project_CH.tasks.manager_based.locomotion.mdp.rewards import (
    body_relative_height_gated,
    rew_approach_ee_object,
    rew_carry_stability,
    rew_grasp_soft,
    rew_nav_to_object,
    rew_nav_to_zone,
    rew_place_release,
    rew_place_soft,
    pen_drop,
    pen_premature_close,
)

from isaaclab.envs import mdp


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
                func=get_object_pose_b,
                params={"object": SceneEntityCfg(name="object_box")},
            ),
        )
        setattr(
            self.observations.policy,
            "wbc_dest_pose_b",
            ObsTerm(
                func=get_dest_pose_b,
                params={"dest": SceneEntityCfg(name="place_zone")},
            ),
        )
        setattr(
            self.observations.policy,
            "wbc_ee_pose_b",
            ObsTerm(
                func=get_ee_pose_b,
                params={
                    "ee": SceneEntityCfg(
                        name="robot", body_names=["piper_gripper_base"]
                    )
                },
            ),  # 필요 시 6으로
        )
        setattr(
            self.observations.policy,
            "wbc_grip_open",
            ObsTerm(
                func=get_gripper_opening,
            ),
        )

        # concat 모드가 꺼져 있으면 켜주기 (term들을 하나의 벡터로 이어붙임)
        self.observations.policy.concatenate_terms = True

        # --- terrain/spacing 등 기존 설정 ---
        self.scene.ee_cam = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.scene.env_spacing = 5.0

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)

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

        self.scene.object_box.init_state.pos = (-3.7, -0.25, 0.2)
        self.scene.place_zone.init_state.pos = (1.5, 0.45, 0.01)

        self.action_schema = {
            "base_cmd": {"start": 0, "dim": 3},
            "arm_delta": {"start": 3, "dim": 4},
            "gripper": {"start": 7, "dim": 1},
        }

        # --- Stage-1 locomotion pretrain  ---

        # self.stage_mode = "pretrain"

        # --- Stage-2 wbc ---
        self.stage_mode = "wbc"
        # EE/OBJ/ZONE 엔티티 참조
        ee_cfg = SceneEntityCfg(name="robot", body_names=["piper_gripper_base"])
        obj_cfg = SceneEntityCfg(name="object_box")
        zone_cfg = SceneEntityCfg(name="place_zone")

        # wholebody_env_cfg.py :: __post_init__ 맨 끝 무렵
        stage = getattr(self, "stage_mode", "pretrain")
        if stage == "wbc":  # == "wbc"
            # Rough에서 가져온 추종 보상 무게를 WBC에서 약화
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
            self.rewards.track_ang_vel_z_exp.weight = 0.0
            self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
            setattr(
                self.observations.policy,
                "velocity_commands",
                ObsTerm(
                    func=obs_zero_pad,
                    params={"dim": 3},
                ),
            )

            self.rewards.arm_pos_penalty = RewardTermCfg(
                func=mdp.joint_pos_limits,
                weight=-0.005,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        joint_names="piper_.*",  # 파이퍼 관련 관절만
                    )
                },
            )

            self.rewards.arm_vel_penalty = RewardTermCfg(
                func=mdp.joint_vel_l2,
                weight=-0.001,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        joint_names="piper_.*",
                    )
                },
            )

            # gripper 관련 관절만
            self.rewards.gripper_torque_penalty = RewardTermCfg(
                func=mdp.joint_torques_l2,
                weight=-0.005,
                params={
                    "asset_cfg": SceneEntityCfg(
                        name="robot",
                        joint_names="piper_joint7|piper_joint8",
                    )
                },
            )

            self.rewards.nav_to_object = RewardTermCfg(
                func=rew_nav_to_object,
                weight=2.0,
            )
            self.rewards.nav_to_zone = RewardTermCfg(
                func=rew_nav_to_zone,
                weight=2.0,
            )

            # 0) 접근
            self.rewards.approach = RewardTermCfg(
                func=rew_approach_ee_object,
                weight=2.5,
                params={
                    "ee_cfg": ee_cfg,
                    "object_cfg": obj_cfg,
                    "dist_scale": 0.06,
                    "use_base_frame": True,
                },
            )

            # 1) 조기닫힘 페널티
            self.rewards.pen_premature_close = RewardTermCfg(
                func=pen_premature_close,
                weight=1.0,  # 내부에 weight가 있으니 여기선 1.0 유지 권장
                params={
                    "ee_cfg": ee_cfg,
                    "object_cfg": obj_cfg,
                    "close_thresh": 0.02,
                    "far_dist": 0.15,
                },
            )

            # 2) 그랩
            self.rewards.grasp_soft = RewardTermCfg(
                func=rew_grasp_soft,
                weight=5.0,
                params={
                    "ee_cfg": ee_cfg,
                    "object_cfg": obj_cfg,
                    "ee_obj_dist_scale": 0.06,
                    "grip_close_scale": 0.01,
                },
            )

            # 3) 운반 안정성
            self.rewards.carry_stability = RewardTermCfg(
                func=rew_carry_stability,
                weight=4.0,
                params={
                    "ee_cfg": ee_cfg,
                    "object_cfg": obj_cfg,
                    "dist_scale": 0.06,
                    "ang_vel_penalty": 0.05,
                },
            )

            # 4) 드랍 페널티
            self.rewards.pen_drop = RewardTermCfg(
                func=pen_drop,
                weight=1.0,  # 내부 weight=3.0 곱해져서 꽤 크게 작용
                params={
                    "ee_cfg": ee_cfg,
                    "object_cfg": obj_cfg,
                    "open_thresh": 0.03,
                    "far_dist": 0.20,
                    "z_fall_delta": 0.06,
                },
            )

            # 5) 플레이스(접근)
            self.rewards.place_soft = RewardTermCfg(
                func=rew_place_soft,
                weight=5.0,
                params={
                    "object_cfg": obj_cfg,
                    "zone_cfg": zone_cfg,
                    "zone_half_size_xy": (0.25, 0.25),
                    "height_tolerance": 0.08,
                },
            )

            # 6) 릴리즈(안에서 벌리면 + / 밖이면 -)
            self.rewards.place_release = RewardTermCfg(
                func=rew_place_release,
                weight=2.0,
                params={
                    "object_cfg": obj_cfg,
                    "zone_cfg": zone_cfg,
                    "zone_half_size_xy": (0.25, 0.25),
                    "open_increase": 0.01,
                },
            )

        self.rewards.body_height.func = body_relative_height_gated
