from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Image Classification API"
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # 모델 관련 설정
    MODEL_PATH: str = "checkpoints/20250123_124210/best_model.onnx"
    CLASS_MAPPING_PATH: str = "data/class_mapping.json"

    class Config:
        case_sensitive = True


settings = Settings()
