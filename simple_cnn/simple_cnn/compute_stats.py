from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def compute_stats():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    train_dir = Path(__file__).parent / config['data']['processed_dir'] / "train"

    dataset = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
            transforms.ToTensor(),
        ])
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0

    print(f"Computing stats over {len(dataset)} images")

    for imgs, _ in loader:
        # imgs: [B, C, H, W]
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, 3, -1) # [B, C, H*W]
        mean += imgs.mean(dim=2).sum(dim=0)
        std  += imgs.std(dim=2).sum(dim=0)
        n    += batch_size

    mean /= n
    std  /= n

    print(f"\nResults:")
    print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"\nPaste into config.yaml:")
    print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")


if __name__ == "__main__":
    compute_stats()