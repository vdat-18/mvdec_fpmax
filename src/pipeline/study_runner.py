"""Run one (dataset, algorithm) group of the FP-Max study and write long-format CSV.

Every FP-Max grid setting is its own arm on every report seed; nothing is selected
from the grid, so the whole distribution of outcomes is reported.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.clustering import labels_hash
from pipeline.external_metrics import compute_external_metrics
from pipeline.reference_evaluation import (
    ReferenceSpace,
    build_reference_distance,
    score_internal,
)
from pipeline.study_arms import ALGORITHMS, STATUS_OK, ArmFit, fit_arm
from pipeline.study_config import StudyConfig
from pipeline.study_data import StudyData
from pipeline.study_ffs import forward_select_itemsets
from pipeline.study_itemsets import (
    STATUS_TOO_MANY,
    FpmaxSetting,
    ItemsetBlock,
    build_itemset_block,
)

STUDY_COLUMNS = (
    "dataset",
    "backbone",
    "algorithm",
    "arm",
    "seed",
    "strategy",
    "n_bins",
    "min_support",
    "n_itemsets",
    "itemset_signature",
    "n_selected",
    "status",
    "error",
    "acc",
    "nmi",
    "ari",
    "ref_silhouette",
    "ref_davies_bouldin",
    "ref_calinski_harabasz",
    "ffs_candidates",
    "fit_seconds",
    "labels_hash",
)
STATUS_SKIPPED_AFTER_TOO_MANY = "skipped_after_too_many_itemsets"
STATUS_BASELINE_UNAVAILABLE = "baseline_unavailable"
STATUS_FFS_BASELINE_INVALID = "ffs_baseline_invalid"


def build_grid_blocks(
    data: StudyData,
    study: StudyConfig,
) -> dict[FpmaxSetting, ItemsetBlock]:
    """Mine every grid setting once, in grid order.

    Supports run high to low for each (strategy, bins) pair. Once a setting exceeds
    the itemset cap, lower supports are not mined (their count and mining time only
    grow) and are recorded with status ``skipped_after_too_many_itemsets``. Data with
    no numeric attributes uses a single (strategy, bins) pair.
    """

    blocks: dict[FpmaxSetting, ItemsetBlock] = {}
    # Binning only affects numeric attributes; without any, every strategy and bin
    # count mines identical itemsets, so only the first pair is kept.
    has_numeric = data.numeric.shape[1] > 0
    strategies = study.fpmax_strategies if has_numeric else study.fpmax_strategies[:1]
    bin_counts = study.fpmax_n_bins if has_numeric else study.fpmax_n_bins[:1]
    for strategy in strategies:
        for n_bins in bin_counts:
            skipping = False
            for support in study.fpmax_min_supports:
                setting = FpmaxSetting(strategy, n_bins, support)
                if skipping:
                    blocks[setting] = ItemsetBlock(
                        pd.DataFrame(index=range(data.n_rows)),
                        STATUS_SKIPPED_AFTER_TOO_MANY,
                        0,
                    )
                    continue
                block = build_itemset_block(data, setting, study.max_itemsets)
                blocks[setting] = block
                skipping = block.status == STATUS_TOO_MANY
    return blocks


def itemset_signature(block: ItemsetBlock) -> str:
    """Short hash of the itemset columns, so identical blocks can be deduplicated."""

    names = "\n".join(sorted(block.columns.columns))
    return hashlib.sha256(names.encode("utf-8")).hexdigest()[:12]


def _evaluate(
    data: StudyData,
    reference: ReferenceSpace,
    labels: np.ndarray | None,
) -> dict:
    """External metrics (only when true labels exist) and reference-space scores."""

    if labels is None:
        return {}
    scores = score_internal(reference, labels)
    values = {
        "ref_silhouette": scores.silhouette,
        "ref_davies_bouldin": scores.davies_bouldin,
        "ref_calinski_harabasz": scores.calinski_harabasz,
        "labels_hash": labels_hash(labels),
    }
    if data.labels is not None:
        external = compute_external_metrics(
            data.labels,
            labels,
            n_clusters=data.n_clusters,
            allow_missing_clusters=True,
        )
        values.update(acc=external.acc, nmi=external.nmi, ari=external.ari)
    return values


def _row(
    data: StudyData,
    algorithm: str,
    arm: str,
    seed: int | None,
    status: str,
    *,
    setting: FpmaxSetting | None = None,
    **values,
) -> dict:
    row = dict.fromkeys(STUDY_COLUMNS, np.nan)
    row.update(
        dataset=data.name,
        backbone=data.backbone,
        algorithm=algorithm,
        arm=arm,
        seed=seed,
        status=status,
        error="",
    )
    if setting is not None:
        row.update(
            strategy=setting.strategy,
            n_bins=setting.n_bins,
            min_support=setting.min_support,
        )
    row.update(values)
    return row


def _fit_row(
    data: StudyData,
    reference: ReferenceSpace,
    algorithm: str,
    arm: str,
    seed: int,
    fit: ArmFit,
    *,
    setting: FpmaxSetting | None = None,
    **values,
) -> dict:
    return _row(
        data,
        algorithm,
        arm,
        seed,
        fit.status,
        setting=setting,
        error=fit.error or "",
        fit_seconds=fit.fit_seconds,
        **_evaluate(data, reference, fit.labels),
        **values,
    )


def _write_outputs(
    rows: list[dict],
    saved_labels: dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    pd.DataFrame(rows, columns=list(STUDY_COLUMNS)).to_csv(csv_path, index=False)
    np.savez_compressed(out_dir / f"{stem}.labels.npz", **saved_labels)
    return csv_path


def done_marker_path(csv_path: Path) -> Path:
    """Marker file written when a group finished; a bare CSV may be partial."""

    return csv_path.with_suffix(".done")


def shard_suffix(seeds: tuple[int, ...]) -> str:
    """File-name suffix identifying a seed shard."""

    return "__seeds-" + "-".join(str(seed) for seed in seeds)


def _setting_key(setting: FpmaxSetting) -> str:
    return f"{setting.strategy}-{setting.n_bins}-{setting.min_support}"


def _run_setting(
    data: StudyData,
    algorithm: str,
    study: StudyConfig,
    reference: ReferenceSpace,
    setting: FpmaxSetting,
    block: ItemsetBlock,
    none_fits: dict[int, ArmFit],
    rows: list[dict],
    saved_labels: dict[str, np.ndarray],
) -> None:
    """Add the fpmax and fpmax_ffs rows of one grid setting for every report seed."""

    key = _setting_key(setting)
    shared = {
        "setting": setting,
        "n_itemsets": block.n_itemsets,
        "itemset_signature": itemset_signature(block),
    }
    for seed in study.report_seeds:
        fit = fit_arm(data, algorithm, block.columns, seed, study)
        rows.append(_fit_row(data, reference, algorithm, "fpmax", seed, fit, **shared))
        if fit.labels is not None:
            saved_labels[f"fpmax__{key}__{seed}"] = fit.labels

        baseline = none_fits[seed]
        if baseline.labels is None:
            rows.append(
                _row(
                    data,
                    algorithm,
                    "fpmax_ffs",
                    seed,
                    STATUS_BASELINE_UNAVAILABLE,
                    **shared,
                )
            )
            continue
        ffs = forward_select_itemsets(
            data, algorithm, block.columns, seed, study, baseline.labels
        )
        ffs_fit = ArmFit(
            labels=ffs.labels,
            status=STATUS_OK if ffs.baseline_valid else STATUS_FFS_BASELINE_INVALID,
            cost=float("nan"),
            error=None,
            fit_seconds=float("nan"),
            init=None,
        )
        rows.append(
            _fit_row(
                data,
                reference,
                algorithm,
                "fpmax_ffs",
                seed,
                ffs_fit,
                n_selected=len(ffs.selected),
                ffs_candidates=ffs.n_candidates_evaluated,
                **shared,
            )
        )
        saved_labels[f"fpmax_ffs__{key}__{seed}"] = ffs.labels


def run_group(
    data: StudyData,
    algorithm: str,
    study: StudyConfig,
    out_dir: Path,
    stem_suffix: str = "",
) -> Path:
    """Run every arm on every report seed and write the CSV plus a labels archive.

    ``stem_suffix`` distinguishes the files of seed shards of one (dataset, algorithm).
    The seedless status rows of settings without itemsets repeat in every shard; the
    results loader drops those duplicates.

    Arms: ``none`` (no itemsets) and, per grid setting, ``fpmax`` (all itemsets) and
    ``fpmax_ffs`` (forward-selected itemsets). ``kmeans`` only has ``none``. A setting
    that mined nothing, too much, or was skipped gets one seedless row with its
    status. Files are rewritten after each stage so a later failure keeps earlier rows.
    """

    if algorithm not in ALGORITHMS:
        msg = f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}."
        raise ValueError(msg)
    reference = build_reference_distance(
        data.reference_df,
        metric=data.reference_metric,
    )
    out_dir = Path(out_dir)
    stem = f"{data.name}__{algorithm}{stem_suffix}"
    rows: list[dict] = []
    saved_labels: dict[str, np.ndarray] = {}

    none_fits: dict[int, ArmFit] = {}
    for seed in study.report_seeds:
        fit = fit_arm(data, algorithm, None, seed, study)
        none_fits[seed] = fit
        rows.append(_fit_row(data, reference, algorithm, "none", seed, fit))
        if fit.labels is not None:
            saved_labels[f"none__{seed}"] = fit.labels
    _write_outputs(rows, saved_labels, out_dir, stem)

    if algorithm != "kmeans":
        blocks = build_grid_blocks(data, study)
        for setting, block in blocks.items():
            if block.status != "ok":
                rows.append(
                    _row(
                        data,
                        algorithm,
                        "fpmax",
                        None,
                        block.status,
                        setting=setting,
                        n_itemsets=block.n_itemsets,
                    )
                )
                continue
            _run_setting(
                data,
                algorithm,
                study,
                reference,
                setting,
                block,
                none_fits,
                rows,
                saved_labels,
            )
            _write_outputs(rows, saved_labels, out_dir, stem)

    csv_path = _write_outputs(rows, saved_labels, out_dir, stem)
    none_rows = [r for r in rows if r["arm"] == "none"]
    if none_rows and all(r["status"] == "error" for r in none_rows):
        # Every baseline fit failed: a systemic problem (config or code), not data.
        msg = f"All {algorithm} fits on {data.name} failed: {none_rows[0]['error']}"
        raise RuntimeError(msg)
    # The CSV is rewritten after every stage, so only this marker proves completion.
    done_marker_path(csv_path).write_text("done", encoding="utf-8")
    return csv_path
