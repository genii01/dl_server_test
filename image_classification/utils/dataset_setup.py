import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)


class DatasetSetup:
    def __init__(
        self,
        dataset_name: str = "beans",  # 간단한 식물 병해 분류 데이터셋
        data_dir: str = "data",
        train_size: int = 100,  # 클래스당 학습 이미지 수
        val_size: int = 10,  # 클래스당 검증 이미지 수
    ):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.train_size = train_size
        self.val_size = val_size
        # split 이름 매핑 추가
        self.split_names = {
            "train": "train",
            "val": "validation",  # validation을 val로 매핑
        }

    def setup(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        데이터셋 다운로드 및 구조화

        Returns:
            data_paths: 학습/검증용 이미지 경로와 레이블
            class_names: 클래스 이름 목록
        """
        # 데이터셋 로드
        dataset = load_dataset(self.dataset_name)

        # 디렉토리 생성
        self._create_directories()

        # 데이터 구조화
        return self._organize_data(dataset)

    def _create_directories(self):
        """디렉토리 구조 생성"""
        # 기존 데이터 삭제
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)

        # 디렉토리 생성
        for split in ["train", "val"]:
            (self.data_dir / split).mkdir(parents=True)

    def _organize_data(self, dataset) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        데이터셋 구조화 및 파일 복사

        Returns:
            data_paths: {
                'train_paths': [...],
                'train_labels': [...],
                'val_paths': [...],
                'val_labels': [...]
            }
            class_names: 클래스 이름 목록
        """
        # 클래스 정보 추출
        class_names = dataset["train"].features["labels"].names

        # 클래스 인덱스와 이름을 매핑하는 딕셔너리 생성 및 JSON 저장
        class_mapping = {idx: name for idx, name in enumerate(class_names)}
        with open(self.data_dir / "class_mapping.json", "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        data_paths = {
            "train_paths": [],
            "train_labels": [],
            "val_paths": [],
            "val_labels": [],
        }

        # CSV 데이터를 위한 리스트
        csv_data = {
            "train": {"image_path": [], "class_name": []},
            "val": {"image_path": [], "class_name": []},
        }

        # 학습/검증 데이터 분할 및 저장
        for local_split, remote_split in self.split_names.items():
            split_size = self.train_size if local_split == "train" else self.val_size

            # 클래스별 처리
            for class_idx, class_name in enumerate(class_names):
                # 해당 클래스의 이미지만 필터링
                class_images = [
                    item
                    for item in dataset[remote_split]
                    if item["labels"] == class_idx
                ]

                # 무작위 샘플링
                selected_images = random.sample(
                    class_images, min(split_size, len(class_images))
                )

                # 이미지 저장 및 경로 기록
                for idx, item in enumerate(selected_images):
                    image = item["image"]
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image)

                    # 이미지 저장
                    image_path = self.data_dir / local_split / f"{class_name}_{idx}.jpg"
                    image.save(image_path)

                    # 경로 및 레이블 기록
                    data_paths[f"{local_split}_paths"].append(str(image_path))
                    data_paths[f"{local_split}_labels"].append(class_idx)

                    # CSV 데이터 추가
                    csv_data[local_split]["image_path"].append(str(image_path))
                    csv_data[local_split]["class_name"].append(class_name)

                logger.info(
                    f"Processed {len(selected_images)} images for {class_name} ({local_split})"
                )

        # CSV 파일 저장
        for split in ["train", "val"]:
            df = pd.DataFrame(csv_data[split])
            df.to_csv(self.data_dir / f"{split}_dataset.csv", index=False)
            logger.info(f"Saved {split} dataset CSV file")

        return data_paths, class_names


def setup_dataset() -> Tuple[Dict[str, List[str]], List[str]]:
    """
    데이터셋 설정 헬퍼 함수

    Returns:
        data_paths: 이미지 경로와 레이블
        class_names: 클래스 이름 목록
    """
    dataset_setup = DatasetSetup()
    return dataset_setup.setup()
