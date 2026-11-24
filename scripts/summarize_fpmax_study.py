"""Summarise the FP-Max study results into CSV tables."""

import argparse
from pathlib import Path

from pipeline.study_stats import (
    EXTERNAL_METRICS,
    load_results,
    primary_table,
    setting_table,
    stability_table,
    status_table,
)

REFERENCE_METRICS = ("ref_silhouette",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results_dir)

    tables = {
        "status.csv": status_table(results),
        "primary_external.csv": primary_table(results, EXTERNAL_METRICS),
        "primary_reference.csv": primary_table(results, REFERENCE_METRICS),
        "stability.csv": stability_table(args.results_dir, results),
    }
    for metric in (*EXTERNAL_METRICS, *REFERENCE_METRICS):
        tables[f"settings_{metric}.csv"] = setting_table(results, metric)
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False)
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
