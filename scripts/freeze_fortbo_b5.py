#!/usr/bin/env python3
"""Freeze fifteen paired FortBO B5 row documents into one comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from check_fortbo_b5 import BUDGET, SEEDS, check_row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("rows", type=Path, nargs=15)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    references = []
    for path in args.rows:
        row = check_row(path)
        regions = row["configuration"]["regions"]
        references.append({
            "file": os.path.relpath(path, args.output.parent),
            "method": "turbo-m-4" if regions == 4 else "turbo-1",
            "mode": row["problem"]["mode"],
            "seed": row["configuration"]["seed"],
        })
    expected = {
        (method, mode, seed)
        for method, mode in (("turbo-1", "raw"), ("turbo-1", "data-informed"),
                             ("turbo-m-4", "data-informed"))
        for seed in SEEDS
    }
    actual = {(r["method"], r["mode"], r["seed"]) for r in references}
    if actual != expected:
        raise ValueError(f"row set is not the paired F3 set: {sorted(actual)}")
    document = {
        "schema_name": "fortbo.b5-value-only-comparison",
        "schema_version": 1,
        "passed": True,
        "configuration": {
            "case_id": "b5_simple_to_build_qi",
            "budget": BUDGET,
            "workers": 8,
            "paired_seeds": list(SEEDS),
        },
        "raw_documents": references,
        "accounting": {
            "truth_calls_per_run": BUDGET,
            "total_truth_calls": 15 * BUDGET,
        },
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
