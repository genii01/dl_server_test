# 기본 이미지로 Python 3.10 사용
FROM python:3.10-slim as python-base

# Python 관련 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYSETUP_PATH="/opt/pysetup"

# 빌더 이미지
FROM python-base as builder-base
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 의존성 설치
WORKDIR $PYSETUP_PATH
COPY pyproject.toml ./

# tomli 설치 후 의존성 추출 및 설치
RUN pip install tomli \
    && python3 -c "import tomli; deps = tomli.load(open('pyproject.toml', 'rb'))['project']['dependencies']; print('\n'.join(d.split(' ')[0].strip('\"') for d in deps))" > requirements.txt

# 런타임 이미지
FROM python-base as production

# 필요한 시스템 패키지 설치
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 의존성 설치
COPY --from=builder-base $PYSETUP_PATH/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    && rm /requirements.txt

# 애플리케이션 코드 복사
COPY . /app/
WORKDIR /app

# 임시 파일 저장을 위한 디렉토리 생성
RUN mkdir -p /app/temp && chmod 777 /app/temp

# 모델과 데이터를 위한 볼륨 마운트 포인트 생성
VOLUME ["/app/checkpoints", "/app/data"]

# 헬스체크를 위한 포트 노출
EXPOSE 8000

# 실행 명령어
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 