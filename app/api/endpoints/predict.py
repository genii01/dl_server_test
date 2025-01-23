from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import logging

from app.models.schemas import (
    SinglePredictionResponse,
    BatchPredictionResponse,
    PredictionResult,
    ErrorResponse,
)
from app.services.predictor import PredictorService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# 전역 predictor 서비스 인스턴스
predictor_service = PredictorService(settings.MODEL_PATH, settings.CLASS_MAPPING_PATH)


@router.post(
    "/predict",
    response_model=SinglePredictionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def predict_image(file: UploadFile = File(...)):
    """
    단일 이미지 예측 엔드포인트
    """
    try:
        pred_class, probability, inference_time = await predictor_service.predict_image(
            file
        )

        return SinglePredictionResponse(
            prediction=PredictionResult(class_name=pred_class, probability=probability),
            inference_time=inference_time,
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={400: {"model": ErrorResponse}},
)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    배치 이미지 예측 엔드포인트
    """
    try:
        results, inference_time = await predictor_service.predict_batch(files)

        predictions = [
            PredictionResult(class_name=class_name, probability=prob)
            for class_name, prob in results
        ]

        return BatchPredictionResponse(
            predictions=predictions, inference_time=inference_time
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
