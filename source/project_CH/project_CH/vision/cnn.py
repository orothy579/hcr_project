import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # ✅ Adaptive Pooling → 항상 (6x6)으로 줄이기
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        # 이제 입력 차원이 항상 64*6*6 = 2304 으로 고정됨
        self.fc = nn.Linear(64 * 6 * 6, out_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # (N, 64, 6, 6)
        x = x.view(x.size(0), -1)  # (N, 2304)
        x = self.fc(x)  # (N, out_dim)
        return x
