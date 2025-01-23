import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import onnx
import onnxruntime
import numpy as np
from models.efficient_classifier import EfficientNetClassifier
from utils.device_utils import get_device

logger = logging.getLogger(__name__)


class ONNXConverter:
    def __init__(
        self,
        checkpoint_path: str,
        onnx_path: Optional[str] = None,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        device: Optional[torch.device] = None,
    ):
        """
        PyTorch 모델을 ONNX로 변환하는 컨버터

        Args:
            checkpoint_path: PyTorch 체크포인트 경로
            onnx_path: 저장할 ONNX 모델 경로 (기본값: checkpoint와 같은 디렉토리)
            input_shape: 입력 텐서 shape (batch_size, channels, height, width)
            device: 실행 디바이스
        """
        self.device = device or get_device()
        self.checkpoint_path = Path(checkpoint_path)
        self.input_shape = input_shape

        if onnx_path is None:
            self.onnx_path = (
                self.checkpoint_path.parent / f"{self.checkpoint_path.stem}.onnx"
            )
        else:
            self.onnx_path = Path(onnx_path)

    def convert(self) -> str:
        """
        PyTorch 모델을 ONNX로 변환

        Returns:
            str: 저장된 ONNX 모델 경로
        """
        logger.info(f"Loading PyTorch model from {self.checkpoint_path}")
        self.model, _ = EfficientNetClassifier.load_model(
            str(self.checkpoint_path), self.device
        )
        self.model.eval()

        # 더미 입력 생성
        dummy_input = torch.randn(self.input_shape, device=self.device)

        # ONNX 내보내기
        logger.info(f"Converting to ONNX format...")
        torch.onnx.export(
            self.model,
            dummy_input,
            str(self.onnx_path),
            verbose=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=12,
        )

        # ONNX 모델 검증
        self._verify_onnx_model(dummy_input)

        logger.info(f"ONNX model saved to {self.onnx_path}")
        return str(self.onnx_path)

    def _verify_onnx_model(self, dummy_input: torch.Tensor) -> None:
        """
        변환된 ONNX 모델 검증

        Args:
            dummy_input: 검증용 입력 텐서
        """
        logger.info("Verifying ONNX model...")

        # ONNX 모델 로드 및 검증
        onnx_model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(onnx_model)

        # ONNX Runtime으로 추론 테스트
        ort_session = onnxruntime.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )

        # PyTorch 출력
        with torch.no_grad():
            torch_output = self.model(dummy_input).cpu().numpy()

        # ONNX Runtime 출력
        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # 출력 비교
        np.testing.assert_allclose(
            torch_output,
            ort_output,
            rtol=1e-03,
            atol=1e-05,
            err_msg="PyTorch와 ONNX Runtime의 출력이 일치하지 않습니다!",
        )
        logger.info("✓ PyTorch와 ONNX Runtime 출력이 일치합니다")


def main():
    """
    사용 예시
    """
    # 체크포인트 경로 설정
    checkpoint_path = "checkpoints/20250123_124210/best_model.pth"

    # ONNX 변환
    converter = ONNXConverter(
        checkpoint_path=checkpoint_path,
        input_shape=(1, 3, 224, 224),  # (batch_size, channels, height, width)
    )

    onnx_path = converter.convert()
    logger.info(f"변환 완료: {onnx_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
