"""Audit without-FFS Intuitive Native summary against trial logs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY = Path("output/without_ffs_intuitive_native_results.csv")
TRIALS = Path("output/without_ffs_intuitive_native_trials.csv")
ATOL = 1e-12


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> int:
    summary = pd.read_csv(SUMMARY)
    trials = pd.read_csv(TRIALS)
    summary_ok = summary[summary["status"] == "ok"].copy()
    trials_ok = trials[trials["status"] == "ok"].copy()

    best_idx = trials_ok.groupby("job_index")["silhouette_score"].idxmax()
    trial_best = trials_ok.loc[
        best_idx,
        ["job_index", "trial_index", "stage", "silhouette_score", "mu_param", "gamma", "beta"],
    ].rename(
        columns={
            "trial_index": "best_trial_index",
            "stage": "best_stage",
            "silhouette_score": "trial_best_score",
            "mu_param": "trial_best_mu_param",
            "gamma": "trial_best_gamma",
            "beta": "trial_best_beta",
        }
    )
    merged = summary_ok.merge(trial_best, on="job_index", how="left")

    score_ok = np.isclose(
        merged["silhouette_score"],
        merged["trial_best_score"],
        rtol=0.0,
        atol=ATOL,
    )
    params_ok = (
        np.isclose(merged["mu_param"], merged["trial_best_mu_param"])
        & np.isclose(merged["gamma"], merged["trial_best_gamma"])
        & np.isclose(merged["beta"], merged["trial_best_beta"])
    )
    best_rows = trials[as_bool(trials["is_best"])].copy()
    best_count_by_job = best_rows.groupby("job_index").size()

    score_mismatch = merged[~score_ok]
    param_mismatch = merged[~params_ok]
    invalid_best_count = best_count_by_job[best_count_by_job != 1]
    missing_best = sorted(set(summary_ok["job_index"]) - set(best_rows["job_index"]))

    print("Verification target")
    print(f"  summary: {SUMMARY}")
    print(f"  trials:  {TRIALS}")
    print("\nCounts")
    print(f"  summary rows: {len(summary)}")
    print(f"  summary ok rows: {len(summary_ok)}")
    print(f"  trial rows: {len(trials)}")
    print(f"  trial ok rows: {len(trials_ok)}")
    print(f"  is_best rows: {len(best_rows)}")
    print("\nTrial stages")
    print(trials["stage"].value_counts(dropna=False).to_string())
    print("\nTrial statuses")
    print(trials["status"].value_counts(dropna=False).to_string())
    print("\nTop summary rows")
    print(
        summary_ok.sort_values("silhouette_score", ascending=False)[
            ["job_index", "silhouette_score", "n_selected_features", "mu_param", "gamma", "beta"]
        ].head(10).to_string(index=False)
    )

    failures = []
    if not score_mismatch.empty:
        failures.append(f"score mismatch: {len(score_mismatch)} jobs")
    if not param_mismatch.empty:
        failures.append(f"parameter mismatch: {len(param_mismatch)} jobs")
    if not invalid_best_count.empty:
        failures.append(f"invalid is_best count: {len(invalid_best_count)} jobs")
    if missing_best:
        failures.append(f"missing is_best: {len(missing_best)} jobs")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASSED")
    print("  Summary rows exactly match the max ok trial per job_index.")
    print("  Best mu_param/gamma/beta also match the trial log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
