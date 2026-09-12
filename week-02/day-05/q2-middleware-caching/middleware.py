import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.monotonic()

        response = await call_next(request)

        elapsed = time.monotonic() - start_time

        response.headers["X-Response-Time"] = f"{elapsed:.4f}s"

        print(
            f"{request.method} {request.url.path} {response.status_code}"
        )

        return response