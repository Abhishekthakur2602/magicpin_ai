#!/usr/bin/env python3
"""
Push the entire expanded dataset into a running Vera bot via /v1/context,
then fire a /v1/tick with a handful of real trigger ids and print the
composed actions — a fast way to sanity-check the LLM path end-to-end
without hand-writing curl commands.

Usage:
    python3 push_and_test.py --bot-url http://localhost:8080 --dataset-dir expanded
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request as urlrequest


def post(url: str, payload: dict) -> dict:
    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_json(p: Path):
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-url", default="http://localhost:8080")
    ap.add_argument("--dataset-dir", default="expanded")
    ap.add_argument("--num-triggers", type=int, default=5, help="how many test-pair triggers to tick")
    args = ap.parse_args()

    base = Path(args.dataset_dir)
    bot = args.bot_url.rstrip("/")

    print("Pushing categories...")
    for f in sorted((base / "categories").glob("*.json")):
        payload = load_json(f)
        post(f"{bot}/v1/context", {
            "scope": "category", "context_id": payload["slug"], "version": 1, "payload": payload,
        })

    print("Pushing merchants...")
    for f in sorted((base / "merchants").glob("*.json")):
        payload = load_json(f)
        post(f"{bot}/v1/context", {
            "scope": "merchant", "context_id": payload["merchant_id"], "version": 1, "payload": payload,
        })

    print("Pushing customers...")
    for f in sorted((base / "customers").glob("*.json")):
        payload = load_json(f)
        post(f"{bot}/v1/context", {
            "scope": "customer", "context_id": payload["customer_id"], "version": 1, "payload": payload,
        })

    print("Pushing triggers...")
    for f in sorted((base / "triggers").glob("*.json")):
        payload = load_json(f)
        post(f"{bot}/v1/context", {
            "scope": "trigger", "context_id": payload["id"], "version": 1, "payload": payload,
        })

    test_pairs = json.loads((base / "test_pairs.json").read_text())["pairs"]
    sample_trigger_ids = [p["trigger_id"] for p in test_pairs[: args.num_triggers]]

    print(f"\nFiring /v1/tick with {len(sample_trigger_ids)} triggers...\n")
    result = post(f"{bot}/v1/tick", {
        "now": "2026-04-29T10:00:00Z",
        "available_triggers": sample_trigger_ids,
    })

    for action in result.get("actions", []):
        print(f"[{action['trigger_id']}]")
        print(f"  body:  {action['body']}")
        print(f"  cta:   {action['cta']}  send_as: {action['send_as']}")
        print(f"  why:   {action['rationale']}")
        print()


if __name__ == "__main__":
    main()
