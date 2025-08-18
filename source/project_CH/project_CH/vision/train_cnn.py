import os, re, glob, torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from project_CH.vision.terrain_cnn import TerrainCNN

DATA_DIR = "data/terrain_ds/imgs"
NUM_LEVELS = 6  # 환경에 맞춰 조정 => 환경의 terrain_level이 정수가 아니라 실수로 나옴
EPOCHS = 5
BATCH = 256
LR = 1e-3
OUT = "weights/terrain_cnn.pt"


class ImgLevelDS(Dataset):
    def __init__(self, folder):
        self.paths = glob.glob(os.path.join(folder, "*.png"))
        self.t = transforms.Compose(
            [
                transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # [0,1]
            ]
        )
        self.re = re.compile(r"_(\d+)\.png$")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        x = self.t(img)
        y = int(self.re.search(p).group(1))
        return x, y


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = ImgLevelDS(DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TerrainCNN(num_levels=NUM_LEVELS).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(EPOCHS):
        tot, correct, cnt = 0.0, 0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)  #
            loss = crit(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            tot += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            cnt += x.size(0)
        print(f"epoch {ep+1}: loss={tot/cnt:.4f}, acc={correct/cnt:.3f}")

    torch.save(model.state_dict(), OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
