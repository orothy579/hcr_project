# 환경에서 카메라 데이터를 불러와 CNN 임베딩으로 바꿔주는 함수 모음
# 이 파일 덕분에, 시뮬레이션 속 카메라 → 신경망 임베딩 → RL 정책 입력 흐름이 연결됨.
import torch
import torch.nn.functional as F


def get_rgb_tensor(env, sensor_cfg):
    """
    Isaac Lab 카메라 센서로부터 raw RGB 이미지를 PyTorch tensor로 변환.
    """
    cam = env.scene.sensors[sensor_cfg.name]
    # Isaac Lab CameraSensor: rgb 값이 (N,H,W,4) 또는 (N,H,W,3)일 수 있음
    rgb = cam.data.output["rgb"]  # (N, H, W, 4 or 3)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    # to (N,3,H,W), [0,1]
    rgb = rgb.permute(0, 3, 1, 2).contiguous().float() / 255.0
    return rgb


def rgb_ee_embed(env, sensor_cfg):
    """
    위 tensor를 가져옴
    cnn.py의 VisionEncoder로 통과시켜 (N,64) 벡터로 변환
    학습에는 쓰이지 않고 고정된 Encoder (freeze된 CNN) → 보행 정책만 학습
    """
    from project_CH.vision.cnn import VisionEncoder

    # 1) 입력 추출
    x = get_rgb_tensor(env, sensor_cfg)  # (N,3,96,96)
    # 2) 모델 로드(1회 캐싱)
    if not hasattr(env, "_vision_encoder"):
        enc = VisionEncoder(out_dim=64)
        enc.eval()  # 고정
        for p in enc.parameters():
            p.requires_grad_(False)
        env._vision_encoder = enc.to(x.device)
    # 3) 임베딩
    with torch.no_grad():
        feat = env._vision_encoder(x)  # (N,64)
    return feat
