import os
import time
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException, RestException


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 환경변수에서 Tracking URI를 읽고, 없으면 기본값 사용
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

MODEL_NAME = "IrisClassifier"
ALIAS = "production"


def train_model():
    """학습 데이터를 만들고 모델을 학습한 뒤 (model, acc) 반환"""
    X, y = load_iris(return_X_y=True)

    Xtr, Xte ,ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)

    return model, acc

def log_and_register(model, acc):
    """
    1) MLflow run 생성
    2) metric / model artifact 기록
    3) registry에 등록
    4) alias 설정
    """
    client = MlflowClient()
    mlflow.set_experiment("Default")
    # 1) run 기록
    with mlflow.start_run(run_name="register_iris") as run:
        mlflow.log_metric("acc", acc)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_uri = f"runs:/{run.info.run_id}/model"
        print("model_uri = ", model_uri)

    # 2) registry 등록(비동기일 수 있음)
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"registered: {MODEL_NAME} v{mv.version}")

    # 3) alias 지정 (등록 직후엔 준비 안 되었을 수 있어 간단 리트라이 필요)
    for attempt in range(10):
        try:
            client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
            print(f"✅ alias set: {MODEL_NAME}:{ALIAS} -> v{mv.version}")
            return
        except Exception as e:
            # 아직 레지스트리 메타가 반영되지 않은 경우 대비
            if attempt == 9:
                raise
            print(f"retry alias... ({attempt + 1}/10) err={e}")
            time.sleep(1)

def main():
    print(f"tracking uri = {TRACKING_URI}")

    model, acc = train_model()
    print(f"acc = {acc:.4f}")

    log_and_register(model, acc)


if __name__ == "__main__":
    main()