FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY data.py predict.py ./
COPY artifacts/model.joblib artifacts/model.joblib

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
