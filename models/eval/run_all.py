"""Regenerate every evaluation output in models/eval/out/ from the parquet tables.

Usage (repo root):  .venv/bin/python -m models.eval.run_all
"""

import time

from . import ablation, compare, curves, failures, sensitivity
from .harness import OUT, load_dataset


def main():
    t0 = time.perf_counter()
    (OUT / "report_numbers.json").unlink(missing_ok=True)
    ds = load_dataset()
    for mod in [compare, ablation, sensitivity, curves, failures]:  # failures reads compare's runs.csv
        print(f"\n=== {mod.__name__} ===")
        mod.main(ds)
    print(f"\ndone in {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
