"""Paired statistics for the FP-Max study results (long-format CSVs).

Comparisons are always arm versus ``none`` of the same dataset, algorithm and seed.
No grid setting is ever picked as "best": every distinct set of mined itemsets
(``itemset_signature``) counts once, and the outcome per dataset is the median over
those settings, so the headline statistic cannot be inflated by choosing among them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from pipeline.reference_evaluation import pairwise_ari_stability

KEY = ["dataset", "algorithm"]
ARMS = ("fpmax", "fpmax_ffs")
EXTERNAL_METRICS = ("ari", "nmi")
SETTING_COLUMNS = ["strategy", "n_bins", "min_support"]


def load_results(out_dir: Path) -> pd.DataFrame:
    """Concatenate every shard CSV; drop the seedless status rows repeated per shard."""

    paths = sorted(Path(out_dir).glob("*.csv"))
    if not paths:
        msg = f"No result CSVs found in {out_dir}."
        raise FileNotFoundError(msg)
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    seedless = frame["seed"].isna()
    keep_seedless = frame[seedless].drop_duplicates(
        subset=[*KEY, "arm", *SETTING_COLUMNS, "status"]
    )
    return pd.concat([frame[~seedless], keep_seedless], ignore_index=True)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order (nan stays nan)."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    order = valid[np.argsort(values[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def rank_biserial(deltas: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: +1 if every pair favours the arm."""

    nonzero = deltas[deltas != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def wilcoxon_p(deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when every delta is zero."""

    nonzero = deltas[deltas != 0]
    if len(nonzero) == 0:
        return 1.0
    return float(wilcoxon(nonzero, zero_method="wilcox").pvalue)


def paired_deltas(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-seed ``arm - none`` deltas for every distinct itemset block.

    Only rows with status ``ok`` and a finite metric on both sides enter; settings
    mining the same itemsets are collapsed to the first in grid order.
    """

    ok = results[(results["status"] == "ok") & results["seed"].notna()]
    none = ok[ok["arm"] == "none"][[*KEY, "seed", metric]].rename(
        columns={metric: "baseline"}
    )
    arms = ok[ok["arm"].isin(ARMS)].copy()
    arms = arms.sort_values(
        [*KEY, "arm", "strategy", "n_bins", "min_support"],
        ascending=[True, True, True, True, True, False],
    )
    arms = arms.drop_duplicates(subset=[*KEY, "arm", "itemset_signature", "seed"])
    merged = arms.merge(none, on=[*KEY, "seed"], how="inner")
    merged["delta"] = merged[metric] - merged["baseline"]
    merged = merged[np.isfinite(merged["delta"])]
    return merged[
        [
            *KEY,
            "arm",
            "itemset_signature",
            *SETTING_COLUMNS,
            "n_itemsets",
            "seed",
            "delta",
        ]
    ]


def dataset_level(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, algorithm, arm): median over distinct settings of the seed mean."""

    per_setting = deltas.groupby([*KEY, "arm", "itemset_signature"], as_index=False)[
        "delta"
    ].mean()
    return per_setting.groupby([*KEY, "arm"], as_index=False).agg(
        delta=("delta", "median"),
        n_settings=("delta", "size"),
    )


def primary_table(results: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Wilcoxon over datasets per (algorithm, arm, metric); Holm over all of them."""

    rows = []
    for metric in metrics:
        levels = dataset_level(paired_deltas(results, metric))
        for (algorithm, arm), group in levels.groupby(["algorithm", "arm"]):
            deltas = group["delta"].to_numpy(dtype=float)
            rows.append(
                {
                    "metric": metric,
                    "algorithm": algorithm,
                    "arm": arm,
                    "n_datasets": len(deltas),
                    "median_delta": float(np.median(deltas)),
                    "wins": int((deltas > 0).sum()),
                    "losses": int((deltas < 0).sum()),
                    "p_value": wilcoxon_p(deltas) if len(deltas) >= 2 else np.nan,
                    "effect_rank_biserial": rank_biserial(deltas),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm_adjust(table["p_value"].tolist())
    return table


def setting_table(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Distribution of the seed-mean delta over distinct settings, per dataset."""

    deltas = paired_deltas(results, metric)
    per_setting = deltas.groupby([*KEY, "arm", "itemset_signature"], as_index=False)[
        "delta"
    ].mean()
    return per_setting.groupby([*KEY, "arm"], as_index=False).agg(
        n_settings=("delta", "size"),
        share_positive=("delta", lambda values: float((values > 0).mean())),
        median_delta=("delta", "median"),
        min_delta=("delta", "min"),
        max_delta=("delta", "max"),
    )


def stability_table(out_dir: Path, results: pd.DataFrame) -> pd.DataFrame:
    """Mean pairwise ARI between seeds for each arm and distinct itemset block."""

    rows = []
    ok = results[(results["status"] == "ok") & results["seed"].notna()]
    signatures = (
        ok[ok["arm"].isin(ARMS)]
        .sort_values(
            ["strategy", "n_bins", "min_support"], ascending=[True, True, False]
        )
        .drop_duplicates(subset=[*KEY, "arm", "itemset_signature"])
    )
    archives: dict[Path, dict[str, np.ndarray]] = {}

    def labels_for(dataset: str, algorithm: str, key: str, seed: int):
        path = Path(out_dir) / f"{dataset}__{algorithm}__seeds-{seed}.labels.npz"
        if path not in archives:
            archives[path] = dict(np.load(path)) if path.exists() else {}
        return archives[path].get(key)

    for (dataset, algorithm), group in ok.groupby(KEY):
        none_labels = [
            labels_for(dataset, algorithm, f"none__{int(seed)}", int(seed))
            for seed in sorted(group[group["arm"] == "none"]["seed"].unique())
        ]
        none_labels = [labels for labels in none_labels if labels is not None]
        if len(none_labels) >= 2:
            rows.append(
                {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "arm": "none",
                    "itemset_signature": "",
                    "stability_ari": pairwise_ari_stability(none_labels),
                }
            )
    for _, row in signatures.iterrows():
        key_prefix = (
            f"{row['arm']}__{row['strategy']}-{int(row['n_bins'])}-{row['min_support']}"
        )
        seeds = sorted(
            ok[
                (ok["dataset"] == row["dataset"])
                & (ok["algorithm"] == row["algorithm"])
                & (ok["arm"] == row["arm"])
                & (ok["itemset_signature"] == row["itemset_signature"])
            ]["seed"].unique()
        )
        label_sets = [
            labels_for(
                row["dataset"],
                row["algorithm"],
                f"{key_prefix}__{int(seed)}",
                int(seed),
            )
            for seed in seeds
        ]
        label_sets = [labels for labels in label_sets if labels is not None]
        if len(label_sets) >= 2:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "algorithm": row["algorithm"],
                    "arm": row["arm"],
                    "itemset_signature": row["itemset_signature"],
                    "stability_ari": pairwise_ari_stability(label_sets),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["dataset", "algorithm", "arm", "itemset_signature", "stability_ari"],
    )


def status_table(results: pd.DataFrame) -> pd.DataFrame:
    """Row counts by dataset, algorithm, arm and status, so failures stay visible."""

    return (
        results.groupby([*KEY, "arm", "status"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
    )
