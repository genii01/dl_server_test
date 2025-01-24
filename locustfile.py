import os
import random
from typing import List
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner
import logging
import json
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassificationUser(HttpUser):
    # 요청 사이의 대기 시간 (1~5초)
    wait_time = between(1, 5)

    def on_start(self):
        """테스트 시작 시 초기화"""
        # 테스트용 이미지 경로 설정
        self.test_images_dir = Path("data/test_images")
        if not self.test_images_dir.exists():
            raise Exception(f"Test images directory not found: {self.test_images_dir}")

        # 테스트 이미지 목록 캐싱
        self.image_files = [
            f
            for f in self.test_images_dir.glob("*.jpg")
            if f.is_file() and f.stat().st_size < 1024 * 1024  # 1MB 이하 파일만
        ]

        if not self.image_files:
            raise Exception("No test images found")

        logger.info(f"Found {len(self.image_files)} test images")

        # 성능 메트릭 초기화
        self.total_predictions = 0
        self.successful_predictions = 0

    def get_random_images(self, count: int = 1) -> List[Path]:
        """무작위 테스트 이미지 선택"""
        return random.sample(self.image_files, min(count, len(self.image_files)))

    @task(3)  # 가중치 3: 단일 예측이 더 빈번하게 실행
    def predict_single_image(self):
        """단일 이미지 예측 테스트"""
        image_path = random.choice(self.image_files)

        try:
            with open(image_path, "rb") as image_file:
                files = {"file": ("image.jpg", image_file, "image/jpeg")}
                with self.client.post(
                    "/api/v1/predict",
                    files=files,
                    name="/predict (single)",
                    catch_response=True,
                ) as response:
                    if response.status_code == 200:
                        result = response.json()
                        self.successful_predictions += 1
                        response.success()
                        logger.debug(
                            f"Single prediction success: {result['prediction']['class_name']}"
                        )
                    else:
                        response.failure(f"Failed with status {response.status_code}")

        except Exception as e:
            logger.error(f"Single prediction error: {str(e)}")

        self.total_predictions += 1

    @task(1)  # 가중치 1: 배치 예측은 덜 빈번하게 실행
    def predict_batch_images(self):
        """배치 이미지 예측 테스트"""
        batch_size = random.randint(2, 5)  # 2-5개의 이미지로 배치 구성
        image_paths = self.get_random_images(batch_size)

        try:
            files = [
                ("files", (f"image_{i}.jpg", open(path, "rb"), "image/jpeg"))
                for i, path in enumerate(image_paths)
            ]

            with self.client.post(
                "/api/v1/predict/batch",
                files=files,
                name="/predict/batch",
                catch_response=True,
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    self.successful_predictions += len(image_paths)
                    response.success()
                    logger.debug(
                        f"Batch prediction success: {len(result['predictions'])} images"
                    )
                else:
                    response.failure(f"Failed with status {response.status_code}")

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
        finally:
            # 파일 핸들러 정리
            for _, (_, file, _) in files:
                file.close()

        self.total_predictions += len(image_paths)

    def on_stop(self):
        """테스트 종료 시 정리"""
        success_rate = (
            (self.successful_predictions / self.total_predictions * 100)
            if self.total_predictions > 0
            else 0
        )
        logger.info(
            f"Test completed - Success rate: {success_rate:.2f}% ({self.successful_predictions}/{self.total_predictions})"
        )


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Locust 초기화 시 실행"""
    if isinstance(environment.runner, MasterRunner):
        logger.info("Master node initialized")
    elif isinstance(environment.runner, WorkerRunner):
        logger.info("Worker node initialized")
    else:
        logger.info("Local instance initialized")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """테스트 시작 시 실행"""
    logger.info("Starting load test...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """테스트 종료 시 실행"""
    logger.info("Load test completed")
