#!/usr/bin/env python3
"""
Generate submission.jsonl for the 30 canonical test pairs.

Loads the expanded dataset (run dataset/generate_dataset.py first) and calls
compose() directly for each (merchant, trigger, customer?) test pair,
writing one JSON line per pair in the shape the judge's /v1/tick action
objects use, so it's easy to eyeball or diff against live judge_simulator
output.

Usage:
    python3 generate_submission.py --dataset-dir /path/to/expanded --out submission.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from composer import compose


def load_json(p: Path):
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="/home/claude/magicpin/expanded")
    ap.add_argument("--out", default="submission.jsonl")
    args = ap.parse_args()

    base = Path(args.dataset_dir)

    categories = {load_json(f)["slug"]: load_json(f) for f in (base / "categories").glob("*.json")}
    merchants = {load_json(f)["merchant_id"]: load_json(f) for f in (base / "merchants").glob("*.json")}
    customers = {load_json(f)["customer_id"]: load_json(f) for f in (base / "customers").glob("*.json")}
    triggers = {load_json(f)["id"]: load_json(f) for f in (base / "triggers").glob("*.json")}

    test_pairs = json.loads((base / "test_pairs.json").read_text())["pairs"]

    lines = []
    for pair in test_pairs:
        trigger = triggers.get(pair["trigger_id"])
        merchant = merchants.get(pair["merchant_id"])
        customer = customers.get(pair["customer_id"]) if pair.get("customer_id") else None

        if trigger is None or merchant is None:
            lines.append({
                "test_id": pair["test_id"],
                "error": "missing trigger or merchant in dataset",
                "trigger_id": pair["trigger_id"],
                "merchant_id": pair["merchant_id"],
            })
            continue

        category = categories.get(merchant.get("category_slug"))
        result = compose(category, merchant, trigger, customer)

        lines.append({
            "test_id": pair["test_id"],
            "trigger_id": pair["trigger_id"],
            "merchant_id": pair["merchant_id"],
            "customer_id": pair.get("customer_id"),
            "action": "send",
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
