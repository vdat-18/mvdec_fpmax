"""Build auditable row-level and cluster-level interpretation outputs."""

from collections.abc import Iterable

import pandas as pd


def attach_row_mapping(
    assignments: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    assignment_key: str = "original_index",
    mapping_key: str = "csv_row_index",
) -> pd.DataFrame:
    """Join assignments to raw metadata with a complete one-to-one contract."""

    for frame, key, label in (
        (assignments, assignment_key, "assignment"),
        (mapping, mapping_key, "mapping"),
    ):
        if key not in frame.columns:
            msg = f"Missing {label} row key: {key}."
            raise ValueError(msg)
        if frame[key].isna().any():
            msg = f"{label.capitalize()} row key contains missing values: {key}."
            raise ValueError(msg)
        if not frame[key].is_unique:
            msg = f"{label.capitalize()} row key must be unique: {key}."
            raise ValueError(msg)

    assignment_ids = set(assignments[assignment_key])
    mapping_ids = set(mapping[mapping_key])
    if assignment_ids != mapping_ids:
        missing = sorted(assignment_ids - mapping_ids)[:5]
        unexpected = sorted(mapping_ids - assignment_ids)[:5]
        msg = (
            "Assignment and mapping row IDs do not match; "
            f"missing mapping IDs={missing}, unexpected mapping IDs={unexpected}."
        )
        raise ValueError(msg)

    assignment_values = assignments.set_index(assignment_key)
    return mapping.merge(
        assignment_values,
        left_on=mapping_key,
        right_index=True,
        how="left",
        sort=False,
        validate="one_to_one",
    )


def build_cluster_profiles(
    interpretation_df: pd.DataFrame,
    raw_numeric_columns: Iterable[str],
    *,
    cluster_column: str = "artifact_cluster",
) -> pd.DataFrame:
    """Summarize raw numeric features for each cluster without semantic labels."""

    if cluster_column not in interpretation_df.columns:
        msg = f"Missing cluster column: {cluster_column}."
        raise ValueError(msg)

    columns = list(raw_numeric_columns)
    if not columns:
        msg = "At least one raw numeric profile column is required."
        raise ValueError(msg)
    missing = [column for column in columns if column not in interpretation_df.columns]
    if missing:
        msg = f"Missing raw profile columns: {missing}."
        raise ValueError(msg)
    if interpretation_df.empty:
        msg = "Cannot profile an empty interpretation dataset."
        raise ValueError(msg)

    records: list[dict[str, float | int | str]] = []
    total_count = len(interpretation_df)
    for cluster, cluster_df in interpretation_df.groupby(cluster_column, sort=True):
        sample_count = len(cluster_df)
        for feature in columns:
            values = pd.to_numeric(cluster_df[feature], errors="coerce").dropna()
            records.append(
                {
                    "cluster": int(cluster),
                    "feature": feature,
                    "cluster_sample_count": sample_count,
                    "cluster_share": sample_count / total_count,
                    "non_null_count": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "q1": values.quantile(0.25),
                    "median": values.median(),
                    "q3": values.quantile(0.75),
                    "max": values.max(),
                }
            )
    return pd.DataFrame.from_records(records)
