# mlops-mlflow-fastapi

## Run
```bash
docker compose up -d --build --force-recreate
docker compose ps
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:5000/
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/ready
