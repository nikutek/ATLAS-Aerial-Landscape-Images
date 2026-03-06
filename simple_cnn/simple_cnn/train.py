"""
train.py
Training loop for SimpleCNNv2.
All hyperparameters are read from config.yaml.

Usage:
    python train.py
    python train.py --config other_config.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.metrics import precision_score, recall_score
from tqdm.auto import tqdm

from dataset import get_data_loaders
from model import SimpleCNNv2


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = y_pred.argmax(dim=1)
        train_acc += (preds == y).sum().item() / len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(dataloader)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return train_loss / n, train_acc / n, precision, recall


def val_step(model, dataloader, loss_fn, device):
    model.eval()
    val_loss, val_acc = 0, 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

            preds = y_pred.argmax(dim=1)
            val_acc += (preds == y).sum().item() / len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    n = len(dataloader)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_loss / n, val_acc / n, precision, recall


def save_plots(results, run_name, results_dir):
    epochs_range = range(1, len(results['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{run_name}', fontsize=13)

    metrics = [
        ('loss',      'Loss',      axes[0, 0]),
        ('acc',       'Accuracy',  axes[0, 1]),
        ('precision', 'Precision', axes[1, 0]),
        ('recall',    'Recall',    axes[1, 1]),
    ]

    for key, title, ax in metrics:
        ax.plot(epochs_range, results[f'train_{key}'], label='Train')
        ax.plot(epochs_range, results[f'val_{key}'],   label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = results_dir / f'{run_name}_curves.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plots saved: {path}")


def train(config):
    base_dir        = Path(__file__).parent
    checkpoints_dir = base_dir / config['output']['checkpoints_dir']
    results_dir     = base_dir / config['output']['results_dir']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"simplecnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nRun: {run_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    loaders, dataset_sizes, class_names = get_data_loaders(config)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {dataset_sizes['train']} | Val: {dataset_sizes['val']} | Test: {dataset_sizes['test']}\n")

    model = SimpleCNNv2(num_classes=config['model']['num_classes']).to(device)

    cfg_train = config['training']
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg_train['scheduler']['step_size'],
        gamma     = cfg_train['scheduler']['gamma'],
    )

    results = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_precision": [], "val_precision": [],
        "train_recall":    [], "val_recall":    [],
    }

    best_val_acc = 0.0
    save_path    = checkpoints_dir / f"{run_name}.pth"

    for epoch in tqdm(range(cfg_train['epochs']), desc="Training"):
        train_loss, train_acc, train_prec, train_rec = train_step(model, loaders['train'], loss_fn, optimizer, device)
        val_loss,   val_acc,   val_prec,   val_rec   = val_step(model,   loaders['val'],   loss_fn, device)

        scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_precision"].append(val_prec)
        results["val_recall"].append(val_rec)

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            improved = "  *"

        print(f"Epoch {epoch+1:3d} | "
              f"LR: {scheduler.get_last_lr()[0]:.5f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}"
              f"{improved}")

    print(f"\n{'='*55}")
    print(f"Run         : {run_name}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Model saved : {save_path}")
    print(f"{'='*55}")

    history_path = results_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"History saved: {history_path}")

    save_plots(results, run_name, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)