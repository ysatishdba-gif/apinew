import multiprocessing

import app.main
from app.utils.tracing import (
    FlushTracingMiddleware,
    instrument_fastapi,
    setup_tracing,
    shutdown_tracing,
)

# Workers: 2 * CPU + 1, capped at 4 (for 2 vCPU)
cpu_count = multiprocessing.cpu_count()
workers = min(2 * cpu_count + 1, 4)
# UvicornWorker for async FastAPI (threads ignored by this worker class)
worker_class = "uvicorn.workers.UvicornWorker"
# Concurrent connections per worker
worker_connections = 300
# Binding
bind = "0.0.0.0:8000"
# Timeout (LLM calls can be slow)
timeout = 300
# Keepalive (longer for slow LLM requests)
keepalive = 120
# Graceful shutdown
graceful_timeout = 30
# Connection queue (good for burst traffic)
# Reduced to fail fast on rate limits
backlog = 750
# Worker recycling (prevent memory leaks)
max_requests = 1000
max_requests_jitter = 50
# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
# Temp directory
worker_tmp_dir = "/tmp"


def post_fork(server, worker):
    """Initialize tracing in each Gunicorn worker process."""

    tracing_result = setup_tracing()
    if tracing_result:
        tracer, provider = tracing_result
        app.main.tracer = tracer
        instrument_fastapi(app.main.app, tracer_provider=provider)
        app.main.app.add_middleware(FlushTracingMiddleware, tracer_provider=provider)


def worker_exit(server, worker):
    """Flush pending spans when worker exits."""
    shutdown_tracing()
