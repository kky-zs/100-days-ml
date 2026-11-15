import random
import time
from dataclasses import dataclass, asdict # gives a clean config object
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "auto"           # "auto", "cpu" or "cuda"
    ckpt_dir: str = "checkpoints"
    patience: int = 3              # 0 disables early stopping
    use_scheduler: bool = True     # one-cycle lr schedule, helps convergence
    grad_clip: float | None = 1.0  # clips the grad norm, prevents exploding grad
    log_every: int = 1             # how often to print

class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainConfig,
                 criterion=None, optimizer=None, scheduler=None):
        self.cfg = config
        self.device = self._resolve_device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        if scheduler is None and config.use_scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.lr,
                total_steps=config.epochs * len(train_loader),
            )
        self.scheduler = scheduler

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_device(name: str) -> torch.device: # "auto" picks the GPU if available.
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step() # steps once per batch

            bs = y.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += bs
        return total_loss / n, correct / n

    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            va_loss, va_acc = self._run_epoch(self.val_loader, train=False)
            lr = self.optimizer.param_groups[0]["lr"]

            for key, val in zip(self.history, (tr_loss, tr_acc, va_loss, va_acc, lr)):
                self.history[key].append(val)

            improved = va_loss < self.best_val_loss
            if improved:
                self.best_val_loss = va_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint("last.pt", epoch)
            if improved:
                self.save_checkpoint("best.pt", epoch)

            if (epoch + 1) % self.cfg.log_every == 0:
                print(f"epoch {epoch + 1:02d}/{self.cfg.epochs} | "
                      f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                      f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
                      f"lr {lr:.2e} | {time.time() - t0:.1f}s"
                      f"{' *' if improved else ''}")

            if self.cfg.patience and self.epochs_no_improve >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        return self.history

    def evaluate(self, loader):
        return self._run_epoch(loader, train=False)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        all_logits, all_labels = [], []
        for x, y in loader:
            all_logits.append(self.model(x.to(self.device)).cpu())
            all_labels.append(y)
        return torch.cat(all_logits), torch.cat(all_labels)

    def save_checkpoint(self, name: str, epoch: int):
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(), # Adam keeps moving averages
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "epochs_no_improve": self.epochs_no_improve,
            "config": asdict(self.cfg),
        }, Path(self.cfg.ckpt_dir) / name)

    def load_checkpoint(self, name: str):
        ckpt = torch.load(Path(self.cfg.ckpt_dir) / name, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and ckpt["scheduler_state"] is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.history = ckpt["history"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.epochs_no_improve = ckpt["epochs_no_improve"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Loaded {name}: continuing from epoch {self.start_epoch + 1}")