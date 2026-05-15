"""Ask a running GARDIAN server (fast path, no model reload).

Usage:
  .venv/bin/python scripts/11_ask_gardian_live.py --question "..."
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask running GARDIAN server.")
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--question-type", type=str, default="other")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--reader-react", action="store_true")
    p.add_argument("--pretty", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "question": args.question,
        "question_type": args.question_type,
        "reader_react": bool(args.reader_react),
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            if args.pretty:
                print(json.dumps(obj, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(obj, ensure_ascii=False))
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            obj = json.loads(body)
            print(json.dumps(obj, ensure_ascii=False))
        except Exception:
            print(json.dumps({"ok": False, "error": f"HTTP {e.code}: {e.reason}"}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
