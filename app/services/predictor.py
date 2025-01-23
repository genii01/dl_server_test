import logging
from pathlib import Path
from typing import List, Tuple
import time
from fastapi import UploadFile
import numpy as np
from PIL import Image
import io

from app.core.exceptions import InvalidImageException
from image_classification.inference_onnx import InferenceOnnx, load_onnx_model

logger = logging.getLogger(__name__)


class PredictorService:
    def __init__(self, model_path: str, class_mapping_path: str):
        self.model = self._load_model(model_path, class_mapping_path)

    @staticmethod
    def _load_model(model_path: str, class_mapping_path: str) -> InferenceOnnx:
        logger.info("Loading model...")
        return load_onnx_model(model_path, class_mapping_path)

    async def predict_image(self, file: UploadFile) -> Tuple[str, float, float]:
        start_time = time.time()

        try:
            # 이미지 읽기
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            # 임시 파일로 저장
            with Path("temp.jpg").open("wb") as f:
                f.write(contents)

            # 예측
            pred_class, prob = self.model.predict_single("temp.jpg")

            # 임시 파일 삭제
            Path("temp.jpg").unlink()

            inference_time = time.time() - start_time
            return pred_class, float(prob), inference_time

        except Exception as e:
            logger.error(f"Error predicting image: {str(e)}")
            raise InvalidImageException(str(e))

    async def predict_batch(
        self, files: List[UploadFile], batch_size: int = 32
    ) -> Tuple[List[Tuple[str, float]], float]:
        start_time = time.time()

        try:
            # 임시 파일들 저장
            temp_paths = []
            for i, file in enumerate(files):
                contents = await file.read()
                temp_path = f"temp_{i}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(contents)
                temp_paths.append(temp_path)

            # 배치 예측
            results = self.model.predict_batch(temp_paths, batch_size)

            # 임시 파일들 삭제
            for path in temp_paths:
                Path(path).unlink()

            inference_time = time.time() - start_time
            return results, inference_time

        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            # 임시 파일 정리
            for path in temp_paths:
                try:
                    Path(path).unlink()
                except:
                    pass
            raise InvalidImageException(str(e))
