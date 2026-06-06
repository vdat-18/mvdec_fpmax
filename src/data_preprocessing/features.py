"""Feature engineering steps copied from the legacy Tiki preprocessing notebook."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

CONFIDENCE_Z_95 = 1.959963984540054
RAW_FEATURE_COLUMNS = [
    "Followers",
    "Total Revenue",
    "Years Joined",
    "Rating Quality",
    "Good Review",
    "Med Review",
    "Bad Review",
]
MODEL_FEATURE_COLUMNS = {
    "Followers": "followers",
    "Total Revenue": "revenue",
    "Years Joined": "yearsjoined",
    "Rating Quality": "rating",
    "Good Review": "goodreview",
    "Med Review": "mediumreview",
    "Bad Review": "badreview",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error when expected raw columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        msg = f"Missing required columns: {', '.join(missing)}"
        raise ValueError(msg)


def wilson_rating_score(
    average_rating: float,
    total_ratings: float,
    z: float = CONFIDENCE_Z_95,
) -> float:
    """Return the Wilson lower-bound rating quality on the 0-5 scale."""

    if pd.isna(average_rating) or pd.isna(total_ratings) or total_ratings <= 0:
        return 0.0

    p_hat = average_rating / 5
    denominator = 1 + z**2 / total_ratings
    center = p_hat + z**2 / (2 * total_ratings)
    margin = z * np.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * total_ratings)) / total_ratings
    )
    return float((center - margin) / denominator * 5)


def wilson_review_share_score(
    positive_count: float,
    total_count: float,
    z: float = CONFIDENCE_Z_95,
) -> float:
    """Return the Wilson lower-bound score for one review bucket."""

    if pd.isna(positive_count) or pd.isna(total_count) or total_count <= 0:
        return 0.0

    p_hat = positive_count / total_count
    denominator = 1 + z**2 / total_count
    center = p_hat + z**2 / (2 * total_count)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total_count)) / total_count)
    return float((center - margin) / denominator)


def add_years_joined(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Add the notebook's Years Joined feature from Year Joined."""

    require_columns(df, ["Year Joined"])
    out = df.copy()
    out["Years Joined"] = reference_year - out["Year Joined"]
    return out


def add_wilson_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rating-quality and review-bucket Wilson features."""

    require_columns(
        df,
        [
            "Rating",
            "Review Count",
            "Total Review",
            "1 Start",
            "2 Start",
            "3 Start",
            "4 Start",
            "5 Start",
        ],
    )

    out = df.copy()
    out["Rating Quality"] = out.apply(
        lambda row: wilson_rating_score(row["Rating"], row["Review Count"]),
        axis=1,
    )
    out["Rating Quality"] = out["Rating Quality"].fillna(0)

    out["Total Review"] = out["Total Review"].replace(0, 1)
    out["Good Count"] = out["4 Start"] + out["5 Start"]
    out["Med Count"] = out["3 Start"]
    out["Bad Count"] = out["1 Start"] + out["2 Start"]

    out["Good Review"] = out.apply(
        lambda row: wilson_review_share_score(row["Good Count"], row["Total Review"]),
        axis=1,
    )
    out["Med Review"] = out.apply(
        lambda row: wilson_review_share_score(row["Med Count"], row["Total Review"]),
        axis=1,
    )
    out["Bad Review"] = out.apply(
        lambda row: wilson_review_share_score(row["Bad Count"], row["Total Review"]),
        axis=1,
    )

    return out


def select_preprocessing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the seven raw engineered features used by the notebook."""

    require_columns(df, RAW_FEATURE_COLUMNS)
    return df[RAW_FEATURE_COLUMNS].copy()


def rename_to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rename notebook feature columns to the current project schema."""

    require_columns(df, MODEL_FEATURE_COLUMNS)
    return df.rename(columns=MODEL_FEATURE_COLUMNS)
