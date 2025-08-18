# terrain_cnn.py  (교체)
import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainCNN(nn.Module):
    """Terrain level 회귀 모델 (입력: RGB)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 1)  # ← 회귀 출력 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        y = self.fc2(x).squeeze(-1)  # (N,)
        return y

    @torch.no_grad()
    def predict(self, x):
        y = self.forward(x)  # (N,)
        conf = torch.ones_like(y)  # 간단히 1.0(게이팅에 쓰고 싶으면 유지)
        return y, conf
