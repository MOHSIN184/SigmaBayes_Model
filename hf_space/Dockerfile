FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY web_utils.py .
COPY bayes_backend ./bayes_backend
COPY artifacts ./artifacts

EXPOSE 10000

CMD ["python", "app.py"]
