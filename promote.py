# promote.py
import argparse
import json
import time
import urllib.request
import urllib.error

from mlflow.tracking import MlflowClient



def http_json(method: str, url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
        return  json.loads(data) if data else {}


def get_ready(api_base: str) -> dict:
    return http_json("GET", f"{api_base}/ready")


def reload_api(api_base: str, admin_token: str) -> dict:
    headers = {"X-Admin-Token": admin_token}
    return  http_json("POST", f"{api_base}/admin/reload", headers=headers)


def wait_ready_version(api_base: str, expect_version: str, timeout_sec: int = 30, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = get_ready(api_base)
            if str(r.get("model_version")) == str(expect_version) and r.get("status") is True:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version", help="target model version to promote, e.g. 3")
    p.add_argument("--model", default="IrisClassifier")
    p.add_argument("--alias", default="production")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--admin-token", default="dev-only-token")
    p.add_argument("--rollback-on-fail", action="store_true", default=True)
    args = p.parse_args()

    client = MlflowClient()

    # 1) 현재 production 버전 기록 (old)
    old_version = None
    try:
        mv = client.get_model_version_by_alias(args.model, args.alias)
        old_version = str(mv.version)
    except Exception:
        old_version = None

    print(
        f"🔎 current {args.model}:{args.alias} -> v{old_version}" if old_version else f"🔎 current {args.model}:{args.alias} not set")

    # 2) alias 이동
    client.set_registered_model_alias(args.model, args.alias, str(args.version))
    print(f"✅ alias set: {args.model}:{args.alias} -> v{args.version}")

    # 3) API reload
    try:
        resp = reload_api(args.api, args.admin_token)
        print(f"🔁 reload response: {resp}")
    except urllib.error.HTTPError as e:
        print(f"❌ reload failed: HTTP {e.code} {e.read().decode('utf-8', errors='ignore')}")
        resp = None
    except Exception as e:
        print(f"❌ reload failed: {e}")
        resp = None

    # 4) ready 확인(대기)
    ok = wait_ready_version(args.api, str(args.version), timeout_sec=30, interval=0.5)
    if ok:
        print(f"🟢 PROMOTE OK: /ready -> v{args.version}")
        return

    print(f"🔴 PROMOTE FAIL: /ready did not become v{args.version}")

    # 5) 실패 시 원복(롤백) 시도
    if args.rollback_on_fail and old_version is not None:
        print(f"↩️  trying rollback to v{old_version} ...")
        try:
            client.set_registered_model_alias(args.model, args.alias, str(old_version))
            print(f"✅ alias rollback: {args.model}:{args.alias} -> v{old_version}")
            try:
                resp2 = reload_api(args.api, args.admin_token)
                print(f"🔁 reload response(after rollback): {resp2}")
            except Exception as e:
                print(f"⚠️  reload after rollback failed: {e}")

            ok2 = wait_ready_version(args.api, str(old_version), timeout_sec=30, interval=0.5)
            if ok2:
                print(f"🟡 ROLLBACK OK: /ready -> v{old_version}")
            else:
                print(f"⚠️  rollback attempted but /ready not confirmed v{old_version}")
        except Exception as e:
            print(f"❌ rollback failed: {e}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()

