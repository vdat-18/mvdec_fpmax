"""Strict loader for the FP-Max study configuration."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from intuitive_kprototypes import INIT_STRATEGIES
from pipeline.fpmax import BIN_LABELS_BY_SIZE
from pipeline.public_data import PUBLIC_DATASET_SPECS

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_CONFIG_PATH = PROJECT_DIR / "configs" / "fpmax_study.json"
STUDY_CONFIG_SCHEMA_VERSION = 1
# "kmeans" binning is random, which would make FP-Max itemsets non-reproducible.
STUDY_STRATEGIES = ("uniform", "quantile", "kmeans")
INTUITIVE_MU_CONVENTIONS = ("eq26", "swapped")
KPROTOTYPES_INITS = ("huang", "cao")
MAX_RESTARTS = 1000  # restart seeds are seed * 1000 + restart
MAX_SEED = 1_000_000
MIN_FFS_ROWS = 10


@dataclass(frozen=True)
class DatasetIntuitiveParams:
    """Intuitive-K-prototypes parameters for one dataset."""

    beta: float
    mu: float
    gamma: float


@dataclass(frozen=True)
class IntuitiveStudyParams:
    """Intuitive options shared by every arm of a dataset."""

    mu_convention: str
    init_strategy: str
    restarts: int
    max_iter: int
    defaults: DatasetIntuitiveParams
    per_dataset: dict[str, DatasetIntuitiveParams]

    def for_dataset(self, name: str) -> DatasetIntuitiveParams:
        """Return the dataset's parameters, or the defaults for unknown names."""

        return self.per_dataset.get(name, self.defaults)


@dataclass(frozen=True)
class ApplicationSpec:
    """An unlabeled application dataset stored as a numeric CSV."""

    csv_path: str
    n_clusters: int


def dekm_study_name(public_name: str) -> str:
    """Study name of a DEKM text dataset: ``REUTERS`` becomes ``dekm_reuters``."""

    return f"dekm_{public_name.lower()}"


@dataclass(frozen=True)
class DekmSpec:
    """DEKM backbone: study bundles exported from the GPU runs.

    Report seed ``s`` clusters the embedding trained with seed
    ``embedding_seed_offset + s``; every bundle holds the same ``n_rows`` objects.
    """

    bundle_dir: str
    datasets: tuple[str, ...]
    embedding_seed_offset: int
    n_rows: int

    @property
    def study_names(self) -> tuple[str, ...]:
        """Study dataset names, e.g. ``dekm_reuters``."""

        return tuple(dekm_study_name(name) for name in self.datasets)

    def bundle_name(self, study_name: str, report_seed: int) -> str:
        """File name of the bundle a (dataset, report seed) job clusters."""

        if study_name not in self.study_names:
            msg = f"{study_name!r} is not a configured DEKM dataset."
            raise ValueError(msg)
        return f"{study_name}__seed-{self.embedding_seed_offset + report_seed}.npz"


@dataclass(frozen=True)
class StudyConfig:
    """Validated FP-Max study protocol."""

    report_seeds: tuple[int, ...]
    fpmax_strategies: tuple[str, ...]
    fpmax_n_bins: tuple[int, ...]
    fpmax_min_supports: tuple[float, ...]
    max_itemsets: int
    ffs_min_improvement: float
    ffs_max_rows: int
    ffs_max_candidates: int
    ffs_max_steps: int
    ffs_restarts: int
    kprototypes_init_methods: tuple[str, ...]
    intuitive: IntuitiveStudyParams
    uci_data_dir: str
    uci_datasets: tuple[str, ...]
    applications: dict[str, ApplicationSpec]
    dekm: DekmSpec


def _exact_keys(payload: object, expected: set[str], where: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != expected:
        msg = f"Study config {where} must have exactly the fields {sorted(expected)}."
        raise ValueError(msg)
    return payload


def _int_list(values: object, where: str) -> tuple[int, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(v, bool) or not isinstance(v, int) for v in values)
        or len(set(values)) != len(values)
    ):
        msg = f"Study config {where} must be a non-empty list of unique integers."
        raise ValueError(msg)
    return tuple(values)


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"Study config {where} must be a positive integer."
        raise ValueError(msg)
    return value


def _finite_number(value: object, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        msg = f"Study config {where} must be a finite number."
        raise ValueError(msg)
    return float(value)


def _dataset_params(payload: object, where: str) -> DatasetIntuitiveParams:
    raw = _exact_keys(payload, {"beta", "mu", "gamma"}, where)
    beta = _finite_number(raw["beta"], f"{where}.beta")
    mu = _finite_number(raw["mu"], f"{where}.mu")
    gamma = _finite_number(raw["gamma"], f"{where}.gamma")
    if beta <= 1 or not 0 <= mu <= 1 or not 0 <= gamma <= 1:
        msg = f"Study config {where} needs beta > 1 and mu, gamma in [0, 1]."
        raise ValueError(msg)
    return DatasetIntuitiveParams(beta=beta, mu=mu, gamma=gamma)


def _subset(values: object, allowed: tuple[str, ...], where: str) -> tuple[str, ...]:
    if (
        not isinstance(values, list)
        or not values
        or any(value not in allowed for value in values)
        or len(set(values)) != len(values)
    ):
        msg = f"Study config {where} must be a unique non-empty subset of {allowed}."
        raise ValueError(msg)
    return tuple(values)


def _dekm_spec(payload: object) -> DekmSpec:
    raw = _exact_keys(
        payload,
        {"bundle_dir", "datasets", "embedding_seed_offset", "n_rows"},
        "data.dekm",
    )
    datasets = _subset(
        raw["datasets"], tuple(PUBLIC_DATASET_SPECS), "data.dekm.datasets"
    )
    offset = raw["embedding_seed_offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        msg = "Study config data.dekm.embedding_seed_offset must be a non-negative int."
        raise ValueError(msg)
    n_rows = _positive_int(raw["n_rows"], "data.dekm.n_rows")
    if n_rows < MIN_FFS_ROWS:
        msg = f"Study config data.dekm.n_rows must be at least {MIN_FFS_ROWS}."
        raise ValueError(msg)
    return DekmSpec(
        bundle_dir=str(raw["bundle_dir"]),
        datasets=datasets,
        embedding_seed_offset=offset,
        n_rows=n_rows,
    )


def load_study_config(path: Path = DEFAULT_STUDY_CONFIG_PATH) -> StudyConfig:
    """Load and validate the study configuration."""

    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    raw = _exact_keys(
        payload,
        {"schema_version", "seeds", "fpmax", "ffs", "kprototypes", "intuitive", "data"},
        "root",
    )
    if raw["schema_version"] != STUDY_CONFIG_SCHEMA_VERSION:
        msg = f"Unsupported study config schema: {raw['schema_version']!r}."
        raise ValueError(msg)

    seeds = _exact_keys(raw["seeds"], {"report"}, "seeds")
    report = _int_list(seeds["report"], "seeds.report")
    if any(not 0 <= seed <= MAX_SEED for seed in report):
        msg = f"Study config seeds must lie in [0, {MAX_SEED}]."
        raise ValueError(msg)

    fpmax = _exact_keys(
        raw["fpmax"],
        {"strategies", "n_bins", "min_supports", "max_itemsets"},
        "fpmax",
    )
    strategies = _subset(fpmax["strategies"], STUDY_STRATEGIES, "fpmax.strategies")
    n_bins = _int_list(fpmax["n_bins"], "fpmax.n_bins")
    if any(b not in BIN_LABELS_BY_SIZE for b in n_bins):
        msg = "Study config fpmax.n_bins must be a subset of the supported bin sizes."
        raise ValueError(msg)
    supports = fpmax["min_supports"]
    if not isinstance(supports, list) or not supports:
        msg = "Study config fpmax.min_supports must be a non-empty list."
        raise ValueError(msg)
    min_supports = tuple(_finite_number(v, "fpmax.min_supports") for v in supports)
    if any(not 0 < v <= 1 for v in min_supports) or list(min_supports) != sorted(
        set(min_supports), reverse=True
    ):
        # The runner skips lower supports once one exceeds the itemset cap.
        msg = "Study config fpmax.min_supports must be strictly descending in (0, 1]."
        raise ValueError(msg)

    ffs = _exact_keys(
        raw["ffs"],
        {"min_improvement", "max_rows", "max_candidates", "max_steps", "restarts"},
        "ffs",
    )
    min_improvement = _finite_number(ffs["min_improvement"], "ffs.min_improvement")
    if min_improvement < 0:
        msg = "Study config ffs.min_improvement must be non-negative."
        raise ValueError(msg)
    ffs_max_rows = _positive_int(ffs["max_rows"], "ffs.max_rows")
    if ffs_max_rows < MIN_FFS_ROWS:
        msg = f"Study config ffs.max_rows must be at least {MIN_FFS_ROWS}."
        raise ValueError(msg)

    kproto = _exact_keys(raw["kprototypes"], {"init_methods"}, "kprototypes")
    init_methods = _subset(
        kproto["init_methods"], KPROTOTYPES_INITS, "kprototypes.init_methods"
    )

    intuitive = _exact_keys(
        raw["intuitive"],
        {
            "mu_convention",
            "init_strategy",
            "restarts",
            "max_iter",
            "defaults",
            "per_dataset",
        },
        "intuitive",
    )
    if intuitive["mu_convention"] not in INTUITIVE_MU_CONVENTIONS:
        msg = "Study config intuitive.mu_convention must be 'eq26' or 'swapped'."
        raise ValueError(msg)
    if intuitive["init_strategy"] not in INIT_STRATEGIES:
        msg = f"Study config intuitive.init_strategy must be one of {INIT_STRATEGIES}."
        raise ValueError(msg)
    restarts = _positive_int(intuitive["restarts"], "intuitive.restarts")
    if restarts > MAX_RESTARTS:
        msg = f"Study config intuitive.restarts must be at most {MAX_RESTARTS}."
        raise ValueError(msg)
    ffs_restarts = _positive_int(ffs["restarts"], "ffs.restarts")
    if ffs_restarts > restarts:
        msg = "Study config ffs.restarts must not exceed intuitive.restarts."
        raise ValueError(msg)
    per_dataset_raw = intuitive["per_dataset"]
    if not isinstance(per_dataset_raw, dict):
        msg = "Study config intuitive.per_dataset must be an object."
        raise ValueError(msg)

    data = _exact_keys(
        raw["data"], {"uci_data_dir", "uci_datasets", "applications", "dekm"}, "data"
    )
    applications_raw = data["applications"]
    if not isinstance(applications_raw, dict):
        msg = "Study config data.applications must be an object."
        raise ValueError(msg)
    applications = {}
    for name, spec in applications_raw.items():
        spec = _exact_keys(
            spec, {"csv_path", "n_clusters"}, f"data.applications.{name}"
        )
        applications[name] = ApplicationSpec(
            csv_path=str(spec["csv_path"]),
            n_clusters=_positive_int(spec["n_clusters"], f"{name}.n_clusters"),
        )
    uci_datasets = data["uci_datasets"]
    if (
        not isinstance(uci_datasets, list)
        or not uci_datasets
        or any(not isinstance(name, str) for name in uci_datasets)
        or len(set(uci_datasets)) != len(uci_datasets)
    ):
        msg = "Study config data.uci_datasets must be a non-empty list of unique names."
        raise ValueError(msg)
    dekm = _dekm_spec(data["dekm"])
    names = [*uci_datasets, *applications, *dekm.study_names]
    if len(set(names)) != len(names):
        msg = "Study config dataset names must be unique across all sources."
        raise ValueError(msg)
    known = {*names, *PUBLIC_DATASET_SPECS}
    unknown = sorted(set(per_dataset_raw) - known)
    if unknown:
        msg = f"Study config intuitive.per_dataset names unknown datasets: {unknown}."
        raise ValueError(msg)

    return StudyConfig(
        report_seeds=report,
        fpmax_strategies=strategies,
        fpmax_n_bins=n_bins,
        fpmax_min_supports=min_supports,
        max_itemsets=_positive_int(fpmax["max_itemsets"], "fpmax.max_itemsets"),
        ffs_min_improvement=min_improvement,
        ffs_max_rows=ffs_max_rows,
        ffs_max_candidates=_positive_int(ffs["max_candidates"], "ffs.max_candidates"),
        ffs_max_steps=_positive_int(ffs["max_steps"], "ffs.max_steps"),
        ffs_restarts=ffs_restarts,
        kprototypes_init_methods=init_methods,
        intuitive=IntuitiveStudyParams(
            mu_convention=intuitive["mu_convention"],
            init_strategy=intuitive["init_strategy"],
            restarts=restarts,
            max_iter=_positive_int(intuitive["max_iter"], "intuitive.max_iter"),
            defaults=_dataset_params(intuitive["defaults"], "intuitive.defaults"),
            per_dataset={
                name: _dataset_params(value, f"intuitive.per_dataset.{name}")
                for name, value in per_dataset_raw.items()
            },
        ),
        uci_data_dir=str(data["uci_data_dir"]),
        uci_datasets=tuple(str(name) for name in uci_datasets),
        applications=applications,
        dekm=dekm,
    )
