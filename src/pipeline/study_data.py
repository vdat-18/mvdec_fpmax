"""Uniform input for the FP-Max study: one object for every backbone."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.clustering import minmax_scale_block
from pipeline.public_artifacts import load_public_artifact
from pipeline.public_data import TWENTY_NEWS_SKLEARN_SOURCE, load_dekm_public_dataset

BACKBONES = ("raw", "dekm")


@dataclass(frozen=True)
class StudyData:
    """Numeric and categorical blocks, optional labels and the reference data.

    ``numeric`` is min-max scaled to [0, 1] and ``categorical`` holds the original
    categorical attributes as strings; either may have zero columns. ``labels`` are
    true labels used only for external evaluation. ``reference_df`` holds the
    original attributes that define the fixed scoring space, which never contains a
    method's own FP-Max columns. All frames use the index ``0..n-1``.
    """

    name: str
    backbone: str
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    labels: np.ndarray | None
    n_clusters: int
    reference_df: pd.DataFrame
    reference_metric: str

    @property
    def n_rows(self) -> int:
        """Number of objects."""

        return len(self.numeric)


def _scaled_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] == 0:
        return frame.reset_index(drop=True)
    return minmax_scale_block(frame.reset_index(drop=True))


def load_uci_study_data(key: str, data_dir: Path) -> StudyData:
    """Load one paper UCI table from the sandbox manifest."""

    data_dir = Path(data_dir)
    with (data_dir / "manifest.json").open(encoding="utf-8") as file:
        manifest = {row["key"]: row for row in json.load(file)}
    if key not in manifest:
        msg = f"Dataset {key!r} is missing from {data_dir / 'manifest.json'}."
        raise ValueError(msg)
    meta = manifest[key]
    frame = pd.read_csv(data_dir / Path(meta["processed_csv"].replace("\\", "/")))
    numeric = _scaled_numeric(frame[meta["numeric_cols"]])
    categorical = frame[meta["categorical_cols"]].astype(str).reset_index(drop=True)
    reference_df = pd.concat([numeric, categorical], axis=1)
    return StudyData(
        name=key,
        backbone="raw",
        numeric=numeric,
        categorical=categorical,
        labels=frame[meta["target_col"]].astype(str).to_numpy(),
        n_clusters=int(meta["n_clusters"]),
        reference_df=reference_df,
        reference_metric="gower",
    )


def load_application_study_data(
    name: str,
    csv_path: Path,
    n_clusters: int,
) -> StudyData:
    """Load an unlabeled numeric application dataset."""

    frame = pd.read_csv(csv_path)
    numeric = _scaled_numeric(frame)
    return StudyData(
        name=name,
        backbone="raw",
        numeric=numeric,
        categorical=pd.DataFrame(index=numeric.index),
        labels=None,
        n_clusters=int(n_clusters),
        reference_df=numeric,
        reference_metric="gower",
    )


def load_dekm_study_data(
    artifact_path: Path,
    dataset_name: str,
    dataset_root: Path,
    twenty_news_source: str = TWENTY_NEWS_SKLEARN_SOURCE,
) -> StudyData:
    """Load a DEKM embedding as the numeric block, scored against the tf-idf rows."""

    artifact = load_public_artifact(Path(artifact_path))
    if not isinstance(artifact, dict) or "embedding" not in artifact:
        msg = f"DEKM artifact has no embedding: {artifact_path}."
        raise ValueError(msg)
    dataset = load_dekm_public_dataset(
        dataset_name,
        Path(dataset_root),
        twenty_news_source=twenty_news_source,
    )
    embedding = np.asarray(artifact["embedding"], dtype=float)
    if artifact.get("source_sha256") != dataset.source_sha256:
        msg = "DEKM artifact was not produced from the loaded dataset source."
        raise ValueError(msg)
    if len(embedding) != len(dataset.X):
        msg = (
            f"DEKM artifact rows ({len(embedding)}) do not match the dataset "
            f"({len(dataset.X)})."
        )
        raise ValueError(msg)
    numeric = _scaled_numeric(pd.DataFrame(embedding))
    numeric.columns = [f"emb_{index}" for index in range(numeric.shape[1])]
    return StudyData(
        name=dataset_name,
        backbone="dekm",
        numeric=numeric,
        categorical=pd.DataFrame(index=numeric.index),
        labels=np.asarray(dataset.y),
        n_clusters=dataset.spec.n_clusters,
        reference_df=pd.DataFrame(np.asarray(dataset.X, dtype=float)),
        reference_metric="cosine",
    )
