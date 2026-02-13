from fastapi import FastAPI, Header, HTTPException, UploadFile, File, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.exceptions import MlflowException, RestException
import mlflow.sklearn
import numpy as np
from contextlib import asynccontextmanager
from mlflow.tracking import MlflowClient
import threading
import time
import random


is_reloading = False
model_lock = threading.Lock()
model = None
loaded_version = None  # 서버가 현재 들고 있는 모델 버전 (상태) -> 왜 이 위치인지 : 서버가 켜져 있는 동안 계속 유지돼야 하는 값이기 때문
MODEL_URI = "models:/IrisClassifier@production"
mlflow.set_tracking_uri("http://localhost:5000")

class DummyYoloModel:
    def __init__(self):
        # YOLO 가중치 로딩 흉내
        print("🧠 Initializing Dummy YOLO model...")
        time.sleep(5)
        print("🧠 Dummy YOLO model ready")

    def predict(self, image_bytes: bytes):
        # YOLO 추론 흉내
        time.sleep(0.3)

        # 결과 흉내
        return [
            {
                "class": "object",
                "confidence": round(random.uniform(0.5, 0.9), 2),
                "bbox": [100,120,300,350] # x1,y1,x2,y2
            }
        ]


def load_model():
    global model, loaded_version, is_reloading
    is_reloading = True
    try:
        with model_lock:
            try:
                # 1) 먼저 레지스트리에서 alias 버전 확인
                mv = MlflowClient().get_model_version_by_alias("IrisClassifier", "production")

                # 2) 성공했을 때만 모델 준비
                time.sleep(3)
                model = DummyYoloModel()
                loaded_version = mv.version
                print(f"🧷 Loaded model: Dummy YOLO (alias=production, version={mv.version})")

            except RestException as e:
                # Registered Model이 없거나 alias가 없는 경우 여기로 옴
                model = None
                loaded_version = "none"
                print(f"🟠 Model not ready (registry): {e}")

    finally:
        is_reloading = False



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔵 Loading model...")
    load_model()
    if model is None:
        print("🟡 Model NOT loaded (ready=false)")
    else:
        print("🟢 Model loaded (ready=true)")
    yield
    print("🔴 App shutdown")


app = FastAPI(lifespan=lifespan)


### 기존 스텝에서 사용하던 실수 예측 부분 ###
# class PredictRequest(BaseModel):
#     x1: float
#     x2: float
#     x3: float
#     x4: float


@app.get("/ping")
def ping():
    return {"status": "ok", "model_version" : loaded_version}

ADMIN_TOKEN = "dev-only-token"

@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if model is None:
        return {"status": False, "reason": "model_not_loaded"}
    return {"status": True, "model_version": loaded_version}


@app.post("/admin/reload")
def admin_reload(x_admin_token: str | None = Header(default=None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code = 401, detail = "Unauthorized")

    load_model()
    return {
        "status" : "reloaded",
        "model_version" : loaded_version
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if is_reloading:
        raise  HTTPException(status_code=503, detail="Model is reloading. Try again.")
    # 파일을 읽어서 바이트로 가져오기 (현재는 추론 진행 안함)
    image_bytes = await file.read()

    with model_lock:
        detections = model.predict(image_bytes)

    # 지금 단계에서는 "형태만" YOLO 스타일로 반환
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(image_bytes),
        "model_version": loaded_version,
        "detections": detections
    }


