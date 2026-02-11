# mlops-mlflow-fastapi

## Run
docker compose up -d --build --force-recreate
docker compose ps

## Smoke test
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:5000/
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/ready