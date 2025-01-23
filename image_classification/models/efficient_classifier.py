import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional, Dict, Any


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        num_classes: int = 1000,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        """
        EfficientNet 기반 이미지 분류 모델

        Args:
            model_name: EfficientNet 모델 버전 (b0-b7)
            num_classes: 분류할 클래스 수
            pretrained: 사전학습된 가중치 사용 여부
            dropout_rate: 드롭아웃 비율
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # 베이스 모델 로드
        self.model = (
            EfficientNet.from_pretrained(model_name)
            if pretrained
            else EfficientNet.from_name(model_name)
        )

        # 마지막 분류 레이어 수정
        in_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save_model(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        모델 체크포인트 저장

        Args:
            path: 저장 경로
            epoch: 현재 에폭
            optimizer: 옵티마이저 상태
            scheduler: 학습률 스케줄러 상태 (선택)
            metrics: 저장할 메트릭들 (선택)
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": self.model_name,
            "num_classes": self.num_classes,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)

    @classmethod
    def load_model(
        cls, path: str, device: torch.device
    ) -> tuple[nn.Module, Dict[str, Any]]:
        """
        저장된 모델 로드

        Args:
            path: 모델 경로
            device: 실행 디바이스

        Returns:
            model: 로드된 모델
            checkpoint: 체크포인트 정보
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            model_name=checkpoint["model_name"], num_classes=checkpoint["num_classes"]
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        return model, checkpoint
