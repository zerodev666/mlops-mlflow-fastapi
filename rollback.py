# rollback.py
import argparse
import json
import time
import urllib.request

from mlflow.tracking import MlflowClient


def http_json(method: str, url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data) if data else {}


def wait_ready_version(api_base: str, expect_version: str, timeout_sec: int = 30, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = http_json("GET", f"{api_base}/ready")
            if str(r.get("model_version")) == str(expect_version) and r.get("status") is True:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version", help="target version to rollback to, e.g. 2")
    p.add_argument("--model", default="IrisClassifier")
    p.add_argument("--alias", default="production")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--admin-token", default="dev-only-token")
    args = p.parse_args()

    client = MlflowClient()
    client.set_registered_model_alias(args.model, args.alias, str(args.version))
    print(f"✅ alias set: {args.model}:{args.alias} -> v{args.version}")

    resp = http_json(
        "POST",
        f"{args.api}/admin/reload",
        headers={"X-Admin-Token": args.admin_token},
    )
    print(f"🔁 reload response: {resp}")

    ok = wait_ready_version(args.api, str(args.version), timeout_sec=30, interval=0.5)
    if ok:
        print(f"🟢 ROLLBACK OK: /ready -> v{args.version}")
        return

    print(f"🔴 ROLLBACK FAIL: /ready did not become v{args.version}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
