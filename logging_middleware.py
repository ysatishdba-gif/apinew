import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger, severity):
        super().__init__(app)
        self.lg = logger
        self.Severity = severity

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        self.lg.log_struct(
            message="Request start",
            structured_data={
                "method": request.method,
                "path": request.url.path,
            },
            severity="INFO",
        )
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        self.lg.log_struct(
            message="Request end",
            structured_data={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
            severity="INFO",
        )
        return response
