#!/usr/bin/env python3
"""Reproduce the headline metrics from the retained test artefacts.

This script intentionally does not retrain a model. It verifies the released
prediction/probability arrays, recomputes the paper's headline metrics, and
recreates the percentile bootstrap intervals with NumPy Generator seed 42.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

LABELS = ["abusive", "none", "religious", "sexual", "threat"]
EXPECTED = {
    "macro_f1": 0.8224995103973685,
    "weighted_f1": 0.8331552601710309,
    "accuracy": 0.8339252584150544,
    "mcc": 0.7452080257380084,
    "macro_auroc": 0.9625664080440872,
}


def metric_dict(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(5), zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=range(5), zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "macro_auroc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
    }


def bootstrap(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, B: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values = {k: np.empty(B, dtype=float) for k in ("macro_f1", "weighted_f1", "accuracy", "mcc")}
    n = len(y)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        m = metric_dict(y[idx], pred[idx], proba[idx])
        for k in values:
            values[k][b] = m[k]
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in values.items()}


def resolve(path: str | None, candidates: list[str]) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate paths exists: " + ", ".join(candidates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Path to test_pred.npy")
    parser.add_argument("--proba", help="Path to test_proba.npy")
    parser.add_argument("--meta", help="Path to fusion_meta.npz")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", help="Optional output JSON path")
    parser.add_argument("--check", action="store_true", help="Fail if headline metrics differ by >1e-4")
    args = parser.parse_args()

    pred_path = resolve(args.pred, ["outputs/test_pred.npy", "test_pred.npy"])
    proba_path = resolve(args.proba, ["outputs/test_proba.npy", "test_proba.npy"])
    meta_path = resolve(args.meta, ["outputs/fusion_meta.npz", "fusion_meta.npz"])

    pred = np.load(pred_path)
    proba = np.load(proba_path)
    with np.load(meta_path, allow_pickle=False) as meta:
        if "test_y" not in meta:
            raise KeyError("fusion_meta.npz must contain test_y")
        y = meta["test_y"].astype(int)
        script_mask = meta["test_bn"].astype(bool) if "test_bn" in meta else None

    if pred.ndim != 1 or proba.ndim != 2 or proba.shape[1] != 5 or len(pred) != len(y) or len(proba) != len(y):
        raise ValueError(f"Unexpected shapes: pred={pred.shape}, proba={proba.shape}, y={y.shape}")
    if not np.array_equal(pred, proba.argmax(axis=1)):
        print("WARNING: test_pred.npy is not identical to argmax(test_proba.npy)")

    metrics = metric_dict(y, pred, proba)
    result: dict[str, object] = {
        "n_test": int(len(y)),
        "labels": LABELS,
        "metrics": metrics,
        "bootstrap": bootstrap(y, pred, proba, args.bootstrap, args.seed),
        "inputs": {"pred": str(pred_path), "proba": str(proba_path), "meta": str(meta_path)},
    }

    if script_mask is not None:
        result["script_metrics"] = {}
        for name, mask in (("bangla", script_mask), ("romanized", ~script_mask)):
            if mask.sum() == 0:
                continue
            result["script_metrics"][name] = {
                "n": int(mask.sum()),
                "macro_f1_five_class_zero_support": float(
                    f1_score(y[mask], pred[mask], average="macro", labels=range(5), zero_division=0)
                ),
                "macro_f1_supported_classes_only": float(
                    f1_score(y[mask], pred[mask], average="macro", zero_division=0)
                ),
            }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.check:
        errors = {k: (metrics[k], EXPECTED[k]) for k in EXPECTED if abs(metrics[k] - EXPECTED[k]) > 1e-4}
        if errors:
            raise SystemExit(f"Headline metric check failed: {errors}")
        print("PASS: headline metrics agree with the paper reference values within 1e-4")


if __name__ == "__main__":
    main()
