import logging
from typing import Tuple
import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class InferenceOnnx:
    def __init__(self, session, class_mapping, input_size=(224, 224)):
        self.session = session
        self.class_mapping = class_mapping
        self.input_size = input_size

        # 이미지 전처리 변환 설정
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        이미지 전처리

        Args:
            image_path: 이미지 파일 경로

        Returns:
            np.ndarray: 전처리된 이미지 (shape: [1, 3, height, width])
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        # [C, H, W] -> [1, C, H, W] 형태로 변환
        input_tensor = image_tensor.unsqueeze(0).numpy()

        # 입력 shape 확인 및 로깅
        logger.debug(f"Input tensor shape: {input_tensor.shape}")

        return input_tensor

    def predict_single(self, image_path: str) -> Tuple[str, float]:
        """
        단일 이미지 예측

        Args:
            image_path: 이미지 경로

        Returns:
            tuple: (예측 클래스 이름, 확률)
        """
        # 이미지 전처리
        input_tensor = self.preprocess_image(image_path)

        # 입력 shape 검증
        if len(input_tensor.shape) != 4:
            raise ValueError(
                f"Expected 4D input tensor, but got shape {input_tensor.shape}"
            )
        if input_tensor.shape[1:] != (3, *self.input_size):
            raise ValueError(
                f"Expected input shape [batch, 3, {self.input_size[0]}, {self.input_size[1]}], "
                f"but got {input_tensor.shape}"
            )

        # 추론
        outputs = self.session.run(None, {"input": input_tensor})[0]

        # 결과 처리
        probabilities = np.squeeze(outputs)
        softmax_probs = np.exp(probabilities) / np.sum(np.exp(probabilities))
        pred_idx = np.argmax(softmax_probs)
        pred_prob = softmax_probs[pred_idx]

        # 클래스 이름 매핑
        pred_class = self.class_mapping[pred_idx]

        return pred_class, float(pred_prob)

    def predict_batch(
        self, image_paths: list, batch_size: int = 32
    ) -> list[Tuple[str, float]]:
        """
        배치 단위로 다중 이미지 예측

        Args:
            image_paths: 이미지 경로 리스트
            batch_size: 배치 크기 (기본값: 32)

        Returns:
            list[tuple]: (예측 클래스 이름, 확률) 튜플의 리스트
        """
        results = []

        # 배치 단위로 처리
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []

            # 배치 내 이미지들을 전처리
            for img_path in batch_paths:
                tensor = self.preprocess_image(img_path)
                batch_tensors.append(tensor)

            # 배치 텐서 생성 [B, C, H, W]
            input_batch = np.concatenate(batch_tensors, axis=0)

            # 입력 shape 검증
            if input_batch.shape[1:] != (3, *self.input_size):
                raise ValueError(
                    f"Expected input shape [batch, 3, {self.input_size[0]}, {self.input_size[1]}], "
                    f"but got {input_batch.shape}"
                )

            # 배치 추론
            outputs = self.session.run(None, {"input": input_batch})[0]

            # 배치 결과 처리
            for output in outputs:
                softmax_probs = np.exp(output) / np.sum(np.exp(output))
                pred_idx = np.argmax(softmax_probs)
                pred_prob = softmax_probs[pred_idx]
                pred_class = self.class_mapping[pred_idx]
                results.append((pred_class, float(pred_prob)))

            # 진행상황 로깅
            logger.debug(
                f"Processed batch {i//batch_size + 1}, "
                f"images {i+1}-{min(i+batch_size, len(image_paths))}"
            )

        return results


def load_onnx_model(model_path: str, class_mapping_path: str) -> InferenceOnnx:
    """
    ONNX 모델과 클래스 매핑을 로드하여 추론기 생성

    Args:
        model_path: ONNX 모델 파일 경로
        class_mapping_path: 클래스 매핑 JSON 파일 경로

    Returns:
        InferenceOnnx: 추론기 인스턴스
    """
    import json

    # ONNX 모델 로드
    logger.info(f"Loading ONNX model from {model_path}")
    session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    # 클래스 매핑 로드
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    logger.info(f"Loaded {len(class_mapping)} classes")

    return InferenceOnnx(session, class_mapping)


def main():
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # 모델 로드 및 추론기 생성
    model = load_onnx_model(
        "checkpoints/20250123_124210/best_model.onnx", "data/class_mapping.json"
    )

    # 단일 이미지 추론
    image_path = "./data/train/angular_leaf_spot_0.jpg"
    pred_class, prob = model.predict_single(image_path)
    logger.info(f"Predicted class: {pred_class} with probability: {prob:.4f}")

    # 배치 추론 테스트
    image_paths = [
        "./data/train/angular_leaf_spot_0.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        "./data/train/angular_leaf_spot_1.jpg",
        # ... 더 많은 이미지 경로 추가 가능
    ]
    batch_results = model.predict_batch(image_paths, batch_size=2)
    for i, (pred_class, prob) in enumerate(batch_results):
        logger.info(
            f"Image {i+1} - Predicted class: {pred_class} "
            f"with probability: {prob:.4f}"
        )


if __name__ == "__main__":
    main()
