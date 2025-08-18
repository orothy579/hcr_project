import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainCNN(nn.Module):
    """간단한 Terrain Level 분류 CNN (입력: RGB 이미지)"""

    def __init__(self, num_levels=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_levels)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (N,16,H/2,W/2)
        x = F.relu(self.conv2(x))  # (N,32,H/4,W/4)
        x = F.relu(self.conv3(x))  # (N,64,H/8,W/8)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits 출력

    @torch.no_grad()
    def predict(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = probs.max(dim=1).values
        return pred, conf
