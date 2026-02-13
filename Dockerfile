FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지(필요시만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 앱 코드 복사
COPY . /app

# (선택) MLflow tracking URI를 파일스토어로 기본 설정
ENV MLFLOW_TRACKING_URI=file:/app/mlruns
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# uvicorn 실행 (app:app 은 현재 쓰시는 그대로)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]