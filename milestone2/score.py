#!/usr/bin/env python3
"""Scoring script for review-helpfulness classification (CIS 5300 term project).

Usage:
    python score.py --predictions predictions.jsonl --gold test.jsonl

Gold file: one of the test.jsonl files from Milestone 1. Each line is a JSON
object with a boolean field "label" (true = at least 3 helpful votes,
false = 0 helpful votes).

Prediction file: one JSON object per line with
    row_id      0-based index of the record in the gold file
    prediction  0 or 1 (1 = at least 3 helpful votes)
    score       a number; larger means "more likely 1" (used only for ROC-AUC)
    id          optional; if present it must equal the "id" of the gold record

The script prints accuracy (the primary metric), macro-F1 and ROC-AUC.
It uses only the Python standard library.
"""
import argparse
import json
import math
import sys


def fail(msg):
    sys.exit(f"score.py: error: {msg}")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                fail(f"{path}, line {line_no}: not valid JSON ({e})")
    return rows


def to_label(value, where):
    """Map true/false, 1/0, "true"/"false", "1"/"0" to 1/0."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "1"):
        return 1
    if isinstance(value, str) and value.strip().lower() in ("false", "0"):
        return 0
    fail(f"{where}: expected a binary label, got {value!r}")


def load(pred_path, gold_path):
    """Return (gold, pred, score) lists aligned by row_id."""
    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)
    n = len(gold_rows)
    if n == 0:
        fail(f"{gold_path} is empty")
    if len(pred_rows) != n:
        fail(f"{len(pred_rows)} predictions but {n} gold records")

    gold = []
    for i, r in enumerate(gold_rows):
        if "label" not in r:
            fail(f"{gold_path}, record {i}: no 'label' field")
        gold.append(to_label(r["label"], f"{gold_path}, record {i}"))

    pred = [None] * n
    score = [None] * n
    for k, r in enumerate(pred_rows):
        where = f"{pred_path}, line {k + 1}"
        for field in ("row_id", "prediction", "score"):
            if field not in r:
                fail(f"{where}: no '{field}' field")
        i = r["row_id"]
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
            fail(f"{where}: row_id must be an integer in [0, {n - 1}], got {i!r}")
        if pred[i] is not None:
            fail(f"{where}: row_id {i} appears more than once")
        if "id" in r and "id" in gold_rows[i] and r["id"] != gold_rows[i]["id"]:
            fail(f"{where}: id {r['id']!r} does not match gold record {i} "
                 f"({gold_rows[i]['id']!r}); the files are misaligned")
        pred[i] = to_label(r["prediction"], where)
        s = r["score"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
            fail(f"{where}: score must be a finite number, got {s!r}")
        score[i] = float(s)
    return gold, pred, score


def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def f1_for_class(gold, pred, c):
    tp = sum(g == c and p == c for g, p in zip(gold, pred))
    fp = sum(g != c and p == c for g, p in zip(gold, pred))
    fn = sum(g == c and p != c for g, p in zip(gold, pred))
    denom = 2 * tp + fp + fn
    # A class that is never predicted and never occurs has no F1; we score it 0.
    return 2 * tp / denom if denom else 0.0


def macro_f1(gold, pred):
    return (f1_for_class(gold, pred, 0) + f1_for_class(gold, pred, 1)) / 2


def roc_auc(gold, score):
    """Probability that a random positive gets a higher score than a random
    negative, counting ties as 1/2 (Mann-Whitney U / (n_pos * n_neg))."""
    n_pos = sum(gold)
    n_neg = len(gold) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(gold)), key=lambda i: score[i])
    rank_sum_pos = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # tied scores share the average rank
        rank_sum_pos += avg_rank * sum(gold[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, help="prediction JSONL file")
    ap.add_argument("--gold", required=True, help="gold test JSONL file")
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    args = ap.parse_args()

    gold, pred, score = load(args.predictions, args.gold)
    result = {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "roc_auc": roc_auc(gold, score),
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"examples: {result['n']}")
        print(f"accuracy: {result['accuracy']:.4f}")
        print(f"macro_f1: {result['macro_f1']:.4f}")
        print(f"roc_auc:  {result['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
