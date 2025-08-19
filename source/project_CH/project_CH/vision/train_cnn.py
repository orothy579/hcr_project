# train_cnn.py  (회귀 버전, 최소 수정)
import os
import re
import glob
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 프로젝트 경로는 그대로 사용
from project_CH.vision.terrain_cnn import TerrainCNN

DATA_DIR = "data/terrain_ds/imgs"  # imgs/*.png, 파일명 끝이 _<level>.png 여야 함
EPOCHS = 5
BATCH = 256
LR = 1e-3
OUT = "weights/terrain_cnn_reg.pt"  # 회귀 전용 가중치 파일명


class ImgLevelDS(Dataset):
    """
    파일명 패턴: anything_<level>.png
    예) scene_012_3.png  -> y=3.0
    """

    def __init__(self, folder):
        self.paths = glob.glob(os.path.join(folder, "*.png"))
        self.t = transforms.Compose(
            [
                transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # [0,1] 범위
            ]
        )
        # _<정수>.png 를 캡처
        self.re = re.compile(r"_(\d+)\.png$")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        x = self.t(img)
        m = self.re.search(os.path.basename(p))
        if m is None:
            raise ValueError(f"파일명에서 level을 찾을 수 없습니다: {p}")
        y = float(m.group(1))  # 회귀 타겟
        return x, torch.tensor(y, dtype=torch.float32)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    ds = ImgLevelDS(DATA_DIR)
    if len(ds) == 0:
        raise FileNotFoundError(f"학습 이미지가 없습니다: {DATA_DIR}")

    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TerrainCNN().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.SmoothL1Loss()  # 회귀 손실

    model.train()
    for ep in range(1, EPOCHS + 1):
        tot_loss, mae_sum, cnt = 0.0, 0.0, 0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)  # (B,)

            pred = model(x).squeeze(-1)  # (B,)
            loss = crit(pred, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            # 로깅
            bs = x.size(0)
            tot_loss += loss.item() * bs
            mae_sum += torch.abs(pred.detach() - y).mean().item() * bs
            cnt += bs

        print(f"epoch {ep}: loss={tot_loss/cnt:.4f}, mae={mae_sum/cnt:.3f}")

    torch.save(model.state_dict(), OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
