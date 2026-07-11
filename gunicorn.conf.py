"""Gunicorn configuration.

    gunicorn wsgi:app -c gunicorn.conf.py

IMPORTANT — why a single worker:
The app keeps book-generation state in memory (`_book_sessions` in app.py),
runs one background thread per book, and streams progress to the browser over
Server-Sent Events. All of that lives inside a single process. Running more
than one worker would split that state across processes, so a request could
hit a worker that knows nothing about an in-flight book. Scale with THREADS,
not workers.

Override any setting below with an env var (see the `os.environ.get` calls).
"""
import os

# --- Networking -------------------------------------------------------------
# Bind to a UNIX socket behind nginx, or a host:port. Default: localhost only,
# so put a reverse proxy (nginx/caddy) in front for TLS + the public port.
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:5100')

# --- Concurrency ------------------------------------------------------------
workers = 1                      # do not increase — see the note above
worker_class = 'gthread'         # threads handle SSE + background work
threads = int(os.environ.get('GUNICORN_THREADS', '16'))

# --- Timeouts ---------------------------------------------------------------
# SSE connections (/generate/stream) are long-lived; the default 30s timeout
# would kill them mid-book. 0 disables the worker timeout for these threads.
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '0'))
graceful_timeout = 30
keepalive = 5

# --- Logging ----------------------------------------------------------------
accesslog = os.environ.get('GUNICORN_ACCESS_LOG', '-')  # '-' = stdout
errorlog = os.environ.get('GUNICORN_ERROR_LOG', '-')    # '-' = stderr
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')
