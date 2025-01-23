import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    사용 가능한 최적의 디바이스를 반환
    GPU > MPS(Apple Silicon) > CPU 순서로 확인

    Returns:
        torch.device: 사용할 디바이스
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device
