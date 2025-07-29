from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Master 환경에서 로봇 설정 가져오기
from go2_piper_master.tasks.direct.go2_piper_master.go2_piper_master_env_cfg import Go2PiperMasterEnvCfg
from project_CH.tasks.manager_based.locomotion.mdp.rewards import undesired_contacts
from isaaclab.managers import SceneEntityCfg


from go2_piper_master.assets.go2_piper_robot import GO2_PIPER_CFG
from isaaclab.assets import ArticulationCfg


CUSTOM_GO2_PIPER_CFG = GO2_PIPER_CFG.replace(
    spawn=GO2_PIPER_CFG.spawn.replace(merge_fixed_joints=False)
)


@configclass
class Go2PiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CUSTOM_GO2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # 초기화 시 안정적인 자세를 위해 기본 root pose와 joint pos 사용
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.42)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.actuators["base_actuators"].stiffness = 80.0
        self.scene.robot.actuators["base_actuators"].damping = 6.0

        # Height scanner 위치 지정 (base_link 에 부착)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Terrain 스케일 조정 (go2 크기에 맞춤)
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Action scale 조정
        self.actions.joint_pos.scale = 0.25

        self.actions.joint_pos.joint_names = "FL_.*|FR_.*|HL_.*|HR_.*"

        # Push 이벤트 제거 (원하면 활성화 가능)
        self.events.push_robot = None

        # self.events.add_base_mass = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        # COM 위치 무작위화용 이벤트에서도 body 명칭 맞추기
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Reward 설정
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        # 무릎 닿으면 큰 페널티
        self.rewards.undesired_contacts = self.rewards.feet_air_time.__class__(
            func=undesired_contacts,
            weight=-0.3,
            params={
                "sensor_cfg": SceneEntityCfg(
                    name="contact_forces",
                    body_names=".*_thigh|.*_shin"
                )
            }
        )
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Termination 조건 설정
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class Go2PiperRoughEnvCfg_PLAY(Go2PiperRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Play 모드 전용 설정
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
