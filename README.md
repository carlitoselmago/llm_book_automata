# llm_book_automata
Creates books using small LLM models thru LMstudio, publishes every book

## Pasos de procesamiento

- User: sube su takeout
- System: preprocesa takeout
- User: desselecciona datos que no quiere que sean procesados
- System: crea un json de los datos seleccionados finales
- LLM : parsea una ficha de usuario con:
    - Nombre apellidos
    - Género
    - Edad
    - Lugar de nacimiento
    - Intereses []


## Modelos recomendados


## Despliegue en un servidor (producción)

La app es Flask y en producción se sirve con **gunicorn** (no con `python app.py`,
que usa livereload solo para desarrollo).

> ⚠️ **Un solo worker.** El estado de generación de libros vive en memoria, con
> un hilo en segundo plano por libro y streaming por SSE. Todo eso vive en un
> único proceso, así que hay que escalar con **hilos, no con workers**. El
> `gunicorn.conf.py` ya fija `workers = 1` — no lo subas.

### Requisitos del sistema

- Python 3.11+
- Fuentes Liberation (para portada + PDF): en Debian/Ubuntu
  `apt install fonts-liberation2` (se esperan en
  `/usr/share/fonts/truetype/liberation2`).
- Un backend LLM compatible con OpenAI (LM Studio, llama.cpp server, vLLM…)
  accesible desde el servidor.

### Puesta en marcha

```bash
# 1. Código + entorno
cd /opt/llm_book_automata
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # livereload es opcional en el server

# 2. Configuración
cp deploy/env.example /etc/llm-book.env
# edita SECRET_KEY, LM_STUDIO_BASE_URL, OUTPUT_FOLDER, etc.

# 3. Arranque manual para probar
GUNICORN_BIND=127.0.0.1:8000 .venv/bin/gunicorn wsgi:app -c gunicorn.conf.py
```

### Servicio systemd + nginx

Ejemplos listos para usar en `deploy/`:

- `deploy/llm-book.service` → copiar a `/etc/systemd/system/`, ajustar rutas/usuario,
  luego `systemctl enable --now llm-book`.
- `deploy/nginx.conf.example` → proxy inverso; ya trae `client_max_body_size`
  para las subidas grandes de Takeout y el buffering desactivado para el SSE.
- `deploy/env.example` → plantilla de variables de entorno.

### Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `SECRET_KEY` | aleatoria | Clave de sesión estable (cookie del libro sobrevive a reinicios). |
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234` | Backend LLM compatible con OpenAI. |
| `LM_STUDIO_MODEL` | `liquid/lfm2.5-1.2b` | Modelo a usar. |
| `OUTPUT_FOLDER` | `./outputs` | Datos persistentes: biblioteca + PDFs. Móntalo en un volumen. |
| `UPLOAD_FOLDER` | `./uploads` | Zips temporales de Takeout (se borran tras procesar). |
| `MAX_UPLOAD_MB` | `500` | Tamaño máximo de subida. |
| `GUNICORN_BIND` | `127.0.0.1:8000` | host:port o socket UNIX de gunicorn. |
| `GUNICORN_THREADS` | `16` | Concurrencia (usa hilos, no workers). |
| `GUNICORN_TIMEOUT` | `0` | 0 = sin timeout, necesario para el streaming SSE. |