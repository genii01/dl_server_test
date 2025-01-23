from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR


class ModelNotLoadedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Model not loaded"
        )


class InvalidImageException(HTTPException):
    def __init__(self, detail: str = "Invalid image file"):
        super().__init__(status_code=HTTP_400_BAD_REQUEST, detail=detail)
