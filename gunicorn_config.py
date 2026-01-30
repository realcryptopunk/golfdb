# Gunicorn configuration for Golf Swing API

# Server socket
bind = "0.0.0.0:8080"

# Worker processes
workers = 1  # Single worker - two models loaded in memory
worker_class = "sync"
threads = 1
timeout = 300  # 5 min timeout for video processing

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "golfswing-api"

# Server mechanics
daemon = False
preload_app = True  # Load model once, share across workers
