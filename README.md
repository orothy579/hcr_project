# Project CH - Go2+Piper 강화학습 인턴 프로젝트

## 프로젝트 개요

이 프로젝트는 Go2 Quadruped와 Piper Robot Arm을 통합한 로봇 시스템을 위한 강화학습 프로젝트입니다.

### 프로젝트 목표
- 복잡한 지형 극복: 계단, 경사, 장애물이 있는 지형을 안전하게 통과
- 물체 조작: VR 명령에 따른 정밀한 물체 이동 및 조작
- 통합 미션: 지형 극복과 물체 조작을 동시에 수행하는 복합 태스크

### 프로젝트 구조일걸
- 마스터 프로젝트: `go2_piper_master` - 공통 로봇 모델 및 기본 환경 제공
- 인턴 프로젝트: `project_CH` - 마스터를 상속받아 특화된 학습 알고리즘 구현

## 설치 가이드

### 전제조건
- Isaac Lab 설치 완료 (https://isaac-sim.github.io/IsaacLab/main/)
- Python 3.10+ 환경
- CUDA 지원 GPU (권장)

### 1단계: 마스터 프로젝트 설치
```bash
# 프로젝트 루트 디렉토리에서 마스터 프로젝트로 이동
cd go2_piper_master

# 마스터 프로젝트 설치
python -m pip install -e source/go2_piper_master
```

### 2단계: Project CH 설치
```bash
# project_CH 디렉토리로 이동
cd ../project_CH

# Project CH 설치
python -m pip install -e source/project_CH
```

### 3단계: 환경 등록 확인
```bash
# 사용 가능한 환경 목록 확인
python scripts/list_envs.py
```

## 강화학습 훈련

### 기본 훈련
```bash
# 기본 훈련 시작 (기본 환경 사용)
python scripts/rsl_rl/train.py --task=Template-Project-CH-Direct-v0

# 환경 개수 조정 (GPU 성능에 맞게, 기본값: 4096)
python scripts/rsl_rl/train.py --task=Template-Project-CH-Direct-v0 --num_envs=2048

# 최대 반복 횟수 설정
python scripts/rsl_rl/train.py --task=Template-Project-CH-Direct-v0 --max_iterations=2000

# GPU 지정하여 훈련 (멀티 GPU 환경)
python scripts/rsl_rl/train.py --task=Template-Project-CH-Direct-v0 --device=cuda:0
```

### 훈련 모니터링
RSL-RL이 자동으로 `logs/rsl_rl/` 디렉토리에 훈련 로그를 저장합니다.

```bash
# TensorBoard 설치 (처음 한 번만)
pip install tensorboard

# TensorBoard로 실시간 모니터링 (별도 터미널에서 실행)
tensorboard --logdir=logs/rsl_rl/

# 브라우저에서 http://localhost:6006 접속하여 확인
```

### 훈련된 모델 평가
```bash
# 저장된 모델로 평가 실행
python scripts/rsl_rl/play.py --task=Template-Project-CH-Direct-v0 --checkpoint=logs/rsl_rl/EXPERIMENT_NAME/model_XXXX.pt

# 비디오 녹화와 함께 평가
python scripts/rsl_rl/play.py --task=Template-Project-CH-Direct-v0 --checkpoint=PATH_TO_MODEL --video
```

## 환경 수정 방법

### 보상 함수 수정
파일 위치: `source/project_CH/project_CH/tasks/direct/project_ch/project_ch_env_cfg.py`

```python
@configclass
class ProjectChEnvCfg(Go2PiperMasterEnvCfg):
    # CH 전용 보상 스케일 조정 예시
    reward_scale_alive = 1.0
    reward_scale_ch_specific = 0.5  # 새로운 보상 컴포넌트
    
    # 환경 파라미터 수정
    episode_length_s = 20.0  # 에피소드 길이 (초)
    decimation = 4          # 제어 주기
```

### 환경 로직 수정
파일 위치: `source/project_CH/project_CH/tasks/direct/project_ch/project_ch_env.py`

```python
def _compute_ch_specific_rewards(self) -> torch.Tensor:
    """CH 전용 보상 함수 구현"""
    # 여기에 특화된 보상 로직 추가
    reward = torch.zeros(self.num_envs, device=self.device)
    
    # 예시: 높이 유지 보상
    height_reward = torch.exp(-torch.abs(self.robot.data.root_pos_w[:, 2] - 0.4))
    reward += height_reward
    
    return reward
```

### 학습 파라미터 수정
파일 위치: `source/project_CH/project_CH/tasks/direct/project_ch/agents/rsl_rl_ppo_cfg.py`

주요 수정 가능한 파라미터:
- learning_rate: 학습률
- num_learning_epochs: 학습 에포크 수
- mini_batch_size: 미니배치 크기
- clip_param: PPO 클리핑 파라미터

## 문제해결

### 자주 발생하는 문제

1. **마스터 프로젝트 미설치 경고**
```
[WARNING] go2_piper_master not found. Using placeholder configuration.
```
해결방법: 마스터 프로젝트를 먼저 설치하세요.
```bash
cd ../go2_piper_master && python -m pip install -e source/go2_piper_master
```

2. **환경이 목록에 나타나지 않음**
```bash
# Project CH 재설치
python -m pip uninstall project_CH -y
python -m pip install -e source/project_CH
```

3. **GPU 메모리 부족 오류**
```bash
# 환경 개수 줄이기
python scripts/rsl_rl/train.py --task=Template-Project-CH-Direct-v0 --num_envs=1024

# CPU 모드로 실행 (속도 느림)
python scripts/rsl_rl/train.py --task=Template-Project-CH-Direct-v0 --device=cpu
```

### 로그 및 모니터링

1. **훈련 로그 확인**
- 위치: `logs/rsl_rl/` 디렉토리
- TensorBoard: `tensorboard --logdir=logs/rsl_rl/`

2. **시스템 로그**
- Isaac Sim 로그: `~/.nvidia-omniverse/logs/Kit/Isaac-Sim/`

## 마스터 프로젝트와의 관계

- **로봇 모델 공유**: `go2_piper_master.assets.GO2_PIPER_CFG` 사용
- **기본 환경 상속**: `Go2PiperMasterEnv` 클래스를 확장
- **공통 설정 재사용**: 시뮬레이션 파라미터, 관절 설정 등
- **독립적 개발**: 특화된 보상 함수 및 학습 로직 구현

## 도움이 되는 참고 자료

- **Isaac Lab 공식 문서**: https://isaac-sim.github.io/IsaacLab/main/
- **Isaac Lab 튜토리얼**: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html
- **RSL-RL 라이브러리**: https://github.com/leggedrobotics/rsl_rl
- **Go2 공식 저장소**: https://github.com/unitreerobotics
- **Piper 공식 저장소**: https://github.com/agilexrobotics
- **문제해결 가이드**: https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html