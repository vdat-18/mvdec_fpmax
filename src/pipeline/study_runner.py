"""Run one (dataset, algorithm) group of the FP-Max study and write long-format CSV."""

from dataclasses import dataclass
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
    "phase",
    "seed",
    "strategy",
    "n_bins",
    "min_support",
    "n_itemsets",
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
STATUS_NO_SETTING = "no_valid_fpmax_setting"
STATUS_BASELINE_UNAVAILABLE = "baseline_unavailable"


@dataclass(frozen=True)
class SettingChoice:
    """The FP-Max setting picked on selection seeds, with every selection row."""

    setting: FpmaxSetting | None
    rows: list[dict]


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
    phase: str,
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
        phase=phase,
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
    phase: str,
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
        phase,
        seed,
        fit.status,
        setting=setting,
        error=fit.error or "",
        fit_seconds=fit.fit_seconds,
        **_evaluate(data, reference, fit.labels),
        **values,
    )


def select_fpmax_setting(
    data: StudyData,
    algorithm: str,
    study: StudyConfig,
    reference: ReferenceSpace,
    blocks: dict[FpmaxSetting, ItemsetBlock],
) -> SettingChoice:
    """Pick the setting with the best mean reference silhouette on selection seeds.

    Only settings whose fits are valid on every selection seed compete; ties go to
    the earlier grid position. Report seeds are never touched here. True labels are
    only read to fill the (transparency-only) external-metric columns.
    """

    rows: list[dict] = []
    best: tuple[float, FpmaxSetting] | None = None
    for setting, block in blocks.items():
        if block.status != "ok":
            rows.append(
                _row(
                    data,
                    algorithm,
                    "fpmax",
                    "select",
                    None,
                    block.status,
                    setting=setting,
                    n_itemsets=block.n_itemsets,
                )
            )
            continue
        scores: list[float] = []
        for seed in study.selection_seeds:
            fit = fit_arm(data, algorithm, block.columns, seed, study)
            row = _fit_row(
                data,
                reference,
                algorithm,
                "fpmax",
                "select",
                seed,
                fit,
                setting=setting,
                n_itemsets=block.n_itemsets,
            )
            rows.append(row)
            scores.append(
                row["ref_silhouette"] if fit.status == STATUS_OK else float("nan")
            )
        if scores and not any(np.isnan(scores)):
            mean_score = float(np.mean(scores))
            if best is None or mean_score > best[0]:
                best = (mean_score, setting)
    return SettingChoice(setting=None if best is None else best[1], rows=rows)


def run_group(
    data: StudyData,
    algorithm: str,
    study: StudyConfig,
    out_dir: Path,
) -> Path:
    """Run every arm on every report seed and write the CSV plus a labels archive."""

    if algorithm not in ALGORITHMS:
        msg = f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}."
        raise ValueError(msg)
    reference = build_reference_distance(
        data.reference_df,
        metric=data.reference_metric,
    )
    rows: list[dict] = []
    saved_labels: dict[str, np.ndarray] = {}

    none_fits: dict[int, ArmFit] = {}
    for seed in study.report_seeds:
        fit = fit_arm(data, algorithm, None, seed, study)
        none_fits[seed] = fit
        rows.append(_fit_row(data, reference, algorithm, "none", "report", seed, fit))
        if fit.labels is not None:
            saved_labels[f"none__{seed}"] = fit.labels

    if algorithm != "kmeans":
        blocks = build_grid_blocks(data, study)
        choice = select_fpmax_setting(data, algorithm, study, reference, blocks)
        rows.extend(choice.rows)
        for seed in study.report_seeds:
            if choice.setting is None:
                for arm in ("fpmax", "fpmax_ffs"):
                    rows.append(
                        _row(data, algorithm, arm, "report", seed, STATUS_NO_SETTING)
                    )
                continue
            block = blocks[choice.setting]
            fit = fit_arm(data, algorithm, block.columns, seed, study)
            rows.append(
                _fit_row(
                    data,
                    reference,
                    algorithm,
                    "fpmax",
                    "report",
                    seed,
                    fit,
                    setting=choice.setting,
                    n_itemsets=block.n_itemsets,
                )
            )
            if fit.labels is not None:
                saved_labels[f"fpmax__{seed}"] = fit.labels

            baseline = none_fits[seed]
            if baseline.labels is None:
                rows.append(
                    _row(
                        data,
                        algorithm,
                        "fpmax_ffs",
                        "report",
                        seed,
                        STATUS_BASELINE_UNAVAILABLE,
                        setting=choice.setting,
                        n_itemsets=block.n_itemsets,
                    )
                )
                continue
            ffs = forward_select_itemsets(
                data, algorithm, block.columns, seed, study, baseline.labels
            )
            ffs_fit = ArmFit(
                labels=ffs.labels,
                status=STATUS_OK,
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
                    "report",
                    seed,
                    ffs_fit,
                    setting=choice.setting,
                    n_itemsets=block.n_itemsets,
                    n_selected=len(ffs.selected),
                    ffs_candidates=ffs.n_candidates_evaluated,
                )
            )
            saved_labels[f"fpmax_ffs__{seed}"] = ffs.labels

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{data.name}__{algorithm}"
    csv_path = out_dir / f"{stem}.csv"
    pd.DataFrame(rows, columns=list(STUDY_COLUMNS)).to_csv(csv_path, index=False)
    np.savez_compressed(out_dir / f"{stem}.labels.npz", **saved_labels)
    return csv_path
