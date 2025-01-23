# 프로젝트 구조

```
app/
├── __init__.py
├── api/
│   ├── __init__.py
│   ├── endpoints/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   └── predict.py
│   └── router.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── exceptions.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── services/
│   ├── __init__.py
│   └── predictor.py
└── main.py
```

## 서버 실행 방법

다음 명령어를 실행하여 서버를 시작할 수 있습니다:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 이 구현의 주요 특징

1. **계층화된 구조**:
   - API, 서비스, 모델 레이어를 명확히 분리하여 유지보수성과 확장성을 강화했습니다.

2. **Pydantic을 사용한 강력한 타입 검증**:
   - 데이터 유효성 검사 및 스키마 정의에 Pydantic을 활용하여 안전하고 명확한 데이터 처리를 보장합니다.

3. **상세한 에러 처리**:
   - \`core/exceptions.py\` 파일에서 예외를 정의하여 API 전반에 걸쳐 일관된 에러 응답을 제공합니다.

4. **CORS 지원**:
   - 다양한 클라이언트에서 API를 호출할 수 있도록 CORS(Cross-Origin Resource Sharing)를 설정했습니다.

5. **OpenAPI 문서 자동 생성**:
   - FastAPI의 기본 기능으로 엔드포인트에 대한 OpenAPI 문서를 자동으로 생성합니다.

6. **헬스 체크 엔드포인트**:
   - 서버 상태를 빠르게 확인할 수 있는 헬스 체크 엔드포인트를 제공합니다.

7. **단일/배치 예측 지원**:
   - 단일 이미지 예측과 다중 이미지 예측을 모두 지원하는 유연한 API 설계를 구현했습니다.

8. **비동기 처리**:
   - 비동기(async) 프로그래밍을 활용하여 성능과 응답 속도를 최적화했습니다.

9. **로깅**:
   - 서버의 주요 이벤트와 에러를 기록하여 디버깅 및 모니터링에 활용합니다.

## API 엔드포인트

### 1. **헬스 체크**
- **Endpoint**: `GET /api/v1/health`
- **설명**: 서버 상태를 확인합니다.
- **응답 예시**:
  ```json
  {
      "status": "ok"
  }
  ```

### 2. **단일 이미지 예측**
- **Endpoint**: `POST /api/v1/predict`
- **설명**: 단일 이미지에 대한 예측을 수행합니다.
- **요청 예시**:
  ```json
  {
      "image": "base64_encoded_image"
  }
  ```
- **응답 예시**:
  ```json
  {
      "prediction": "class_label",
      "confidence": 0.95
  }
  ```

### 3. **다중 이미지 예측**
- **Endpoint**: `POST /api/v1/predict/batch`
- **설명**: 여러 이미지에 대한 예측을 일괄적으로 수행합니다.
- **요청 예시**:
  ```json
  {
      "images": [
          "base64_encoded_image_1",
          "base64_encoded_image_2"
      ]
  }
  ```
- **응답 예시**:
  ```json
  [
      {
          "prediction": "class_label_1",
          "confidence": 0.92
      },
      {
          "prediction": "class_label_2",
          "confidence": 0.89
      }
  ]
  ```

