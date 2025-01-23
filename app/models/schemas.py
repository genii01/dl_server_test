from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class PredictionResult(BaseModel):
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class SinglePredictionResponse(BaseModel):
    success: bool = True
    prediction: PredictionResult
    inference_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionResponse(BaseModel):
    success: bool = True
    predictions: List[PredictionResult]
    inference_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    timestamp: datetime = Field(default_factory=datetime.now)
