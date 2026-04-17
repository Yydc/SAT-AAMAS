"""Dataset preparation for SAT.

Builds the JSONL files expected by ``configs/sat_*.yaml``. Two modes:

  * ``--dataset demo``  -- writes a synthetic 16-problem arithmetic set so
    the demo config runs end-to-end without any downloads.
  * ``--dataset aime24`` / ``aime25`` / ``math500`` / ``dapo``
                          -- pulls the official benchmark from Hugging Face
    Datasets and converts it into the SAT JSONL schema.

The Hugging Face downloads require the ``datasets`` package; install with
``pip install datasets`` if you plan to reproduce the paper benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAT datasets.")
    parser.add_argument(
        "--dataset",
        choices=["demo", "aime24", "aime25", "math500", "dapo"],
        required=True,
    )
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override default output directory.")
    return parser.parse_args()


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_demo() -> tuple[list[dict], list[dict]]:
    train = [
        {"prompt": f"What is {a} + {b}? Answer with a number.",
         "chosen": str(a + b),
         "rejected": str(a + b + 1)}
        for a in range(1, 5)
        for b in range(1, 5)
    ]
    test = [
        {"problem": f"Compute {a} * {b}.", "answer": str(a * b)}
        for a, b in [(2, 3), (4, 5), (6, 7), (8, 9)]
    ]
    return train, test


def fetch_hf(name: str, split: str, mapping: callable) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The `datasets` package is required for HF downloads.\n"
            "Install it with: pip install datasets"
        ) from exc
    print(f"[hf] downloading {name}:{split} ...")
    ds = load_dataset(name, split=split)
    return [mapping(row) for row in ds]


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data"

    if args.dataset == "demo":
        train, test = build_demo()
        write_jsonl(train, out_root / "demo" / "train.jsonl")
        write_jsonl(test, out_root / "demo" / "test.jsonl")
        print(f"demo set written to {out_root / 'demo'} "
              f"({len(train)} train / {len(test)} test).")
        return

    if args.dataset == "aime24":
        rows = fetch_hf(
            "Maxwell-Jia/AIME_2024",
            "train",
            lambda r: {"problem": r.get("Problem") or r.get("problem"),
                       "answer": str(r.get("Answer") or r.get("answer"))},
        )
        write_jsonl(rows, out_root / "aime24" / "test.jsonl")
        print(f"AIME24 written to {out_root / 'aime24' / 'test.jsonl'} ({len(rows)} problems).")
        return

    if args.dataset == "aime25":
        rows = fetch_hf(
            "yentinglin/aime_2025",
            "train",
            lambda r: {"problem": r.get("problem"), "answer": str(r.get("answer"))},
        )
        write_jsonl(rows, out_root / "aime25" / "test.jsonl")
        print(f"AIME25 written to {out_root / 'aime25' / 'test.jsonl'} ({len(rows)} problems).")
        return

    if args.dataset == "math500":
        rows = fetch_hf(
            "HuggingFaceH4/MATH-500",
            "test",
            lambda r: {"problem": r.get("problem"), "answer": str(r.get("answer"))},
        )
        write_jsonl(rows, out_root / "math500" / "test.jsonl")
        train_rows = fetch_hf(
            "HuggingFaceH4/MATH-500",
            "test",
            lambda r: {"prompt": r.get("problem"),
                       "chosen": str(r.get("answer")),
                       "rejected": ""},
        )
        write_jsonl(train_rows, out_root / "math500" / "train.jsonl")
        print(f"MATH-500 written to {out_root / 'math500'} ({len(rows)} problems).")
        return

    if args.dataset == "dapo":
        rows = fetch_hf(
            "BAAI/DAPO-Math-17k",
            "train",
            lambda r: {"prompt": r.get("prompt") or r.get("problem", ""),
                       "chosen": str(r.get("answer", "")),
                       "rejected": ""},
        )
        write_jsonl(rows, out_root / "dapo" / "train.jsonl")
        print(f"DAPO written to {out_root / 'dapo' / 'train.jsonl'} ({len(rows)} examples).")
        return


if __name__ == "__main__":
    main()
