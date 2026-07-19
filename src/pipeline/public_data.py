"""Load the unmodified public text datasets released with DEKM 2021."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class PublicDatasetSpec:
    """Fixed benchmark contract for one DEKM public dataset."""

    name: str
    title: str
    n_clusters: int
    n_features: int


@dataclass(frozen=True)
class PublicDataset:
    """Canonical unshuffled public dataset and its source fingerprint."""

    spec: PublicDatasetSpec
    X: np.ndarray
    y: np.ndarray
    source_sha256: str


PUBLIC_DATASET_SPECS = {
    "REUTERS": PublicDatasetSpec("REUTERS", "REUTERS-10K", 4, 2000),
    "20NEWS": PublicDatasetSpec("20NEWS", "20NEWS", 20, 2000),
    "RCV1": PublicDatasetSpec("RCV1", "RCV1-10K", 4, 2000),
}


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """Apply the L2 row normalization used by the released DEKM loader."""

    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _load_dense_npz(path: Path) -> np.ndarray:
    """Load either a SciPy sparse NPZ or the dense NPZ files in the release."""

    if not path.exists():
        raise FileNotFoundError(f"Missing DEKM release file: {path}")
    try:
        return sparse.load_npz(path).toarray()
    except ValueError:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                msg = f"Expected one array in DEKM release file: {path}"
                raise ValueError(msg) from None
            return archive[archive.files[0]]


def _read_indices(path: Path) -> np.ndarray:
    """Read one integer index per line from an RCV1 release split file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing DEKM release file: {path}")
    values = [
        int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return np.asarray(values, dtype=np.int64)


def _array_sha256(*arrays: np.ndarray) -> str:
    """Hash array shapes, dtypes, and canonical row-ordered values."""

    digest = sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_reuters(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Reuters arrays distributed by the DEKM code release."""

    x_path = dataset_dir / "10k_feature.npy"
    y_path = dataset_dir / "10k_target.npy"
    if not x_path.exists() or not y_path.exists():
        msg = f"Missing Reuters DEKM release arrays under {dataset_dir}."
        raise FileNotFoundError(msg)
    X = np.load(x_path, allow_pickle=False).astype(np.float32)
    y = np.load(y_path, allow_pickle=False).reshape(-1).astype(np.int32)
    return _normalize_rows(X), y


def _load_20news(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate the train/test arrays from the DEKM release."""

    train_x = _load_dense_npz(dataset_dir / "train_data.npz")
    test_x = _load_dense_npz(dataset_dir / "test_data.npz")
    train_y = _load_dense_npz(dataset_dir / "train_label.npz")
    test_y = _load_dense_npz(dataset_dir / "test_label.npz")
    X = np.concatenate([train_x, test_x]).astype(np.float32)
    y = np.concatenate([train_y, test_y]).reshape(-1).astype(np.int32)
    return _normalize_rows(X), y


def _load_rcv1(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Apply the released DEKM RCV1 split and feature-selection contract."""

    from sklearn.datasets import fetch_rcv1

    test_indices = _read_indices(dataset_dir / "test")
    validation_indices = _read_indices(dataset_dir / "validation")
    try:
        dataset = fetch_rcv1(download_if_missing=False)
    except (OSError, ValueError) as error:
        msg = (
            "The DEKM release stores RCV1 split indices but relies on the local "
            "scikit-learn RCV1 cache for features. The cache is missing or unreadable."
        )
        raise RuntimeError(msg) from error

    selected_indices = np.concatenate([test_indices, validation_indices])
    data = dataset.data[selected_indices]
    target = dataset.target[selected_indices]
    summed_tfidf = np.asarray(data.sum(axis=0)).reshape(-1)
    feature_count = PUBLIC_DATASET_SPECS["RCV1"].n_features
    feature_indices = np.argpartition(-summed_tfidf, feature_count)[:feature_count]
    X = data[:, feature_indices].toarray().astype(np.float32)

    category_names = ("CCAT", "ECAT", "GCAT", "MCAT")
    category_indices = {
        index: class_index
        for class_index, category in enumerate(category_names)
        for index, name in enumerate(dataset.target_names)
        if name == category
    }
    labels: list[int] = []
    for row_index in range(target.shape[0]):
        matching = [
            category_indices[index]
            for index in target[row_index].indices
            if index in category_indices
        ]
        if len(matching) != 1:
            msg = (
                "Each released RCV1 row must have exactly one CCAT/ECAT/GCAT/MCAT "
                f"label; row {row_index} has {len(matching)}."
            )
            raise ValueError(msg)
        labels.append(matching[0])
    return _normalize_rows(X), np.asarray(labels, dtype=np.int32)


def load_dekm_public_dataset(name: str, dataset_root: Path) -> PublicDataset:
    """Load one canonical DEKM public dataset without regenerating its files."""

    normalized_name = name.upper()
    if normalized_name not in PUBLIC_DATASET_SPECS:
        msg = f"Unsupported DEKM public dataset: {name!r}."
        raise ValueError(msg)
    root = Path(dataset_root).resolve()
    dataset_dir = root / normalized_name
    if normalized_name == "REUTERS":
        X, y = _load_reuters(dataset_dir)
    elif normalized_name == "20NEWS":
        X, y = _load_20news(dataset_dir)
    else:
        X, y = _load_rcv1(dataset_dir)

    spec = PUBLIC_DATASET_SPECS[normalized_name]
    if X.ndim != 2 or X.shape[1] != spec.n_features:
        msg = f"{normalized_name}: expected (*, {spec.n_features}), got {X.shape}."
        raise ValueError(msg)
    if len(X) != len(y):
        msg = f"{normalized_name}: X/y row mismatch: {len(X)} != {len(y)}."
        raise ValueError(msg)
    if not np.isfinite(X).all():
        msg = f"{normalized_name}: feature matrix contains NaN or infinite values."
        raise ValueError(msg)
    if len(np.unique(y)) != spec.n_clusters:
        msg = (
            f"{normalized_name}: expected {spec.n_clusters} labels, "
            f"found {len(np.unique(y))}."
        )
        raise ValueError(msg)
    return PublicDataset(
        spec=spec,
        X=X,
        y=y,
        source_sha256=_array_sha256(X, y),
    )
