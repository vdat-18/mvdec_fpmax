"""Uniform input for the FP-Max study: one object for every backbone."""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.clustering import minmax_scale_block
from pipeline.public_artifacts import load_public_artifact
from pipeline.public_data import TWENTY_NEWS_SKLEARN_SOURCE, load_dekm_public_dataset
from pipeline.study_config import dekm_study_name

BACKBONES = ("raw", "dekm", "tfidf")


@dataclass(frozen=True)
class StudyData:
    """Numeric and categorical blocks, optional labels and the reference data.

    ``numeric`` is min-max scaled to [0, 1] for tabular data and kept unscaled for
    DEKM embeddings (DEKM clusters the raw embedding). ``categorical`` holds the
    original categorical attributes as strings; either may have zero columns.
    ``labels`` are true labels used only for external evaluation. ``reference_df``
    holds the original attributes that define the fixed scoring space, which never
    contains a method's own FP-Max columns. All frames use the index ``0..n-1``.
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
    """Load an unlabeled numeric application dataset.

    It is scored with Euclidean distances on its scaled original columns, the
    distance the K-Means baseline uses, so the baseline is never disadvantaged.
    """

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
        reference_metric="euclidean",
    )


DEKM_BUNDLE_KEYS = (
    "embedding",
    "reference",
    "labels",
    "row_index",
    "dataset",
    "n_clusters",
    "source_sha256",
    "embedding_seed",
)


def write_dekm_study_bundle(
    artifact_path: Path,
    dataset_name: str,
    dataset_root: Path,
    out_path: Path,
    n_rows: int,
    row_seed: int = 0,
    twenty_news_source: str = TWENTY_NEWS_SKLEARN_SOURCE,
) -> Path:
    """Save the study rows of one DEKM run as a self-contained ``.npz``.

    DEKM is trained on every row; the study clusters a fixed sample of ``n_rows``
    rows drawn with ``row_seed`` and without labels, so every embedding seed and
    every arm sees the same objects. The bundle holds those rows' embedding, their
    tf-idf vectors (the fixed cosine scoring space) and their true labels, so the
    study needs neither the text sources nor pickles on the machine that runs it.
    """

    artifact = load_public_artifact(Path(artifact_path))
    if not isinstance(artifact, dict) or "embedding" not in artifact:
        msg = f"DEKM artifact has no embedding: {artifact_path}."
        raise ValueError(msg)
    dataset = load_dekm_public_dataset(
        dataset_name,
        Path(dataset_root),
        twenty_news_source=twenty_news_source,
    )
    if artifact.get("source_sha256") != dataset.source_sha256:
        msg = "DEKM artifact was not produced from the loaded dataset source."
        raise ValueError(msg)
    embedding = np.asarray(artifact["embedding"], dtype=np.float32)
    n_total = len(dataset.y)
    if len(embedding) != n_total:
        msg = (
            f"DEKM artifact rows ({len(embedding)}) do not match the dataset "
            f"({n_total})."
        )
        raise ValueError(msg)
    if not 1 <= n_rows <= n_total:
        msg = f"n_rows must lie in [1, {n_total}], got {n_rows}."
        raise ValueError(msg)
    rows = np.sort(
        np.random.default_rng(row_seed).choice(n_total, size=n_rows, replace=False)
    )
    reference = dataset.X[rows]
    if hasattr(reference, "toarray"):
        reference = reference.toarray()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embedding=embedding[rows],
        reference=np.asarray(reference, dtype=np.float32),
        labels=np.asarray(dataset.y)[rows],
        row_index=rows,
        dataset=np.array(dataset.spec.name),
        n_clusters=np.array(int(dataset.spec.n_clusters)),
        source_sha256=np.array(str(dataset.source_sha256)),
        embedding_seed=np.array(int(artifact.get("random_seed") or -1)),
    )
    return out_path


def load_dekm_study_bundle(path: Path, n_rows: int | None = None) -> StudyData:
    """Load a bundle written by ``write_dekm_study_bundle`` (cosine scoring space).

    The embedding is kept unscaled, as DEKM itself clusters it.
    """

    with np.load(Path(path), allow_pickle=False) as bundle:
        missing = sorted(set(DEKM_BUNDLE_KEYS) - set(bundle.files))
        if missing:
            msg = f"DEKM study bundle {path} lacks {missing}."
            raise ValueError(msg)
        values = {key: bundle[key] for key in DEKM_BUNDLE_KEYS}
    embedding = values["embedding"]
    n = len(embedding)
    if len(values["reference"]) != n or len(values["labels"]) != n:
        msg = f"DEKM study bundle {path} has inconsistent row counts."
        raise ValueError(msg)
    if n_rows is not None and n != n_rows:
        msg = f"DEKM study bundle {path} has {n} rows, the config expects {n_rows}."
        raise ValueError(msg)
    numeric = pd.DataFrame(np.asarray(embedding, dtype=float))
    numeric.columns = [f"emb_{index}" for index in range(numeric.shape[1])]
    return StudyData(
        name=dekm_study_name(str(values["dataset"])),
        backbone="dekm",
        numeric=numeric,
        categorical=pd.DataFrame(index=numeric.index),
        labels=np.asarray(values["labels"]),
        n_clusters=int(values["n_clusters"]),
        reference_df=pd.DataFrame(np.asarray(values["reference"], dtype=float)),
        reference_metric="cosine",
    )


def tfidf_backbone(data: StudyData) -> StudyData:
    """The same DEKM study rows, clustered on their tf-idf vectors instead.

    This is the traditional K-Means baseline of the DEKM paper: identical rows,
    labels and scoring space, with the DEKM embedding replaced by the input it
    was learned from.
    """

    if data.backbone != "dekm":
        msg = f"tfidf_backbone needs DEKM study rows, got backbone {data.backbone!r}."
        raise ValueError(msg)
    numeric = data.reference_df.copy()
    numeric.columns = [f"tfidf_{index}" for index in range(numeric.shape[1])]
    return dataclasses.replace(
        data,
        name=data.name.replace("dekm_", "tfidf_", 1),
        backbone="tfidf",
        numeric=numeric,
    )
