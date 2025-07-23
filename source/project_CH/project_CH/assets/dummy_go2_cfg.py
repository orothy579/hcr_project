import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class DummyGo2Cfg(ArticulationCfg):
    """간이 Unitree Go2 로봇 설정 (Piper 부착 없이 기본 Go2)"""

    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/go2/go2.usd",
        scale=(1.0, 1.0, 1.0)
    )

    # 초기 상태 (간단히 바닥 위에 배치)
    init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35), rot=(1.0, 0.0, 0.0, 0.0))

    # 필수 필드: prim_path, actuators (검증 단계에서 필요)
    prim_path: str = "/World/Robot"
    actuators = []  # 기본적으로 빈 리스트로 설정하여 오류 방지