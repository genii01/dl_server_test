import logging
from typing import List, Tuple

import torch
from models.efficient_classifier import EfficientNetClassifier
from PIL import Image
from utils.data_utils import ImageDataset
from utils.device_utils import get_device

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        class_names: List[str] = None,
    ):
        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")
        self.class_names = class_names

        # 모델 로드
        self.model, _ = EfficientNetClassifier.load_model(model_path, self.device)
        self.model.eval()

        # 기본 전처리
        self.transform = ImageDataset._get_default_transform()

    @torch.no_grad()
    def predict_single(self, image_path: str) -> Tuple[str, float]:
        """단일 이미지 예측"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        outputs = self.model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        pred_prob, pred_idx = torch.max(probabilities, dim=1)

        if self.class_names:
            pred_class = self.class_names[pred_idx.item()]
        else:
            pred_class = str(pred_idx.item())

        return pred_class, pred_prob.item()

    @torch.no_grad()
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, float]]:
        """배치 예측"""
        results = []
        for image_path in image_paths:
            pred_class, prob = self.predict_single(image_path)
            results.append((pred_class, prob))
        return results


def main():
    # 예시 사용법
    device = get_device()
    model_path = "checkpoints/20240101_120000/best_model.pth"
    class_names = [...]  # 클래스 이름 리스트

    predictor = Predictor(model_path=model_path, device=device, class_names=class_names)

    # 단일 이미지 예측
    image_path = "path/to/image.jpg"
    pred_class, prob = predictor.predict_single(image_path)
    print(f"Predicted class: {pred_class} with probability: {prob:.4f}")

    # 배치 예측
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    results = predictor.predict_batch(image_paths)
    for path, (pred_class, prob) in zip(image_paths, results):
        print(f"Image: {path}")
        print(f"Predicted class: {pred_class} with probability: {prob:.4f}")


if __name__ == "__main__":
    main()
