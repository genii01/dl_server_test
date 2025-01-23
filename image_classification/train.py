import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from models.efficient_classifier import EfficientNetClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.data_utils import get_data_loaders
from utils.dataset_setup import setup_dataset
from utils.device_utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc="Training") as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with tqdm(self.val_loader, desc="Validation") as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
                )

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, accuracy

    def train(self, num_epochs: int):
        best_acc = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Learning rate 조정
            self.scheduler.step(val_loss)

            logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            # 최고 성능 모델 저장
            if val_acc > best_acc:
                best_acc = val_acc
                self.model.save_model(
                    os.path.join(self.save_dir, "best_model.pth"),
                    epoch,
                    self.optimizer,
                    self.scheduler,
                    {"val_acc": val_acc, "val_loss": val_loss},
                )

            # 주기적 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                self.model.save_model(
                    os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                    epoch,
                    self.optimizer,
                    self.scheduler,
                )


def main():
    # 설정
    device = get_device()
    save_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # 데이터셋 설정
    data_paths, class_names = setup_dataset()
    logger.info(f"Classes: {class_names}")

    # 데이터 로더 생성 - batch_size와 num_workers를 여기서 전달
    train_loader, val_loader = get_data_loaders(
        train_paths=data_paths["train_paths"],
        train_labels=data_paths["train_labels"],
        val_paths=data_paths["val_paths"],
        val_labels=data_paths["val_labels"],
        batch_size=32,  # 여기서 batch_size 설정
        num_workers=4,  # 여기서 num_workers 설정
    )

    # 모델 초기화
    model = EfficientNetClassifier(
        model_name="efficientnet-b0",
        num_classes=len(class_names),
        pretrained=True,
    ).to(device)

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # 트레이너 초기화 및 학습 - batch_size와 num_workers 제거
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
    )

    trainer.train(num_epochs=5)


if __name__ == "__main__":
    main()
