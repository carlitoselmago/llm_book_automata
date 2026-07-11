"""WSGI entrypoint for production servers (gunicorn/uwsgi).

    gunicorn wsgi:app -c gunicorn.conf.py

Exposes the Flask `app` object without importing the dev-only livereload
server (that lives under `if __name__ == '__main__'` in app.py).
"""
from app import app

if __name__ == '__main__':
    app.run()
