"""
Gunicorn configuration for Prompt Detective API on Hetzner CPX31 (8 vCPU, 16GB RAM)
"""

import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
# Hetzner CPX31: 8 vCPU, 16GB RAM
# Using 2 workers × 2 threads = 4 cores per worker
workers = 2
threads = 2
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 5

# Worker process naming
proc_name = "promptscan_x402"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process management
max_requests = 1000
max_requests_jitter = 50

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Debugging
reload = False
spew = False

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment if using HTTPS)
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"
# ssl_version = "TLS"
# cert_reqs = 0
# ca_certs = None
# suppress_ragged_eofs = True
# do_handshake_on_connect = False


# Server hooks
def post_fork(server, worker):
    server.log.info("Worker %s spawned", worker.pid)


def pre_fork(server, worker):
    pass


def pre_exec(server):
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    server.log.info("Server is ready. Spawning workers")


def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")


def worker_abort(worker):
    worker.log.info("Worker received SIGABRT signal")
