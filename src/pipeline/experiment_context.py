"""Dataset-scoped configuration for local MvDEC experiment outputs."""

import json
from dataclasses import dataclass
from pathlib import Path

from config import KPROTOTYPES_N_INIT, OUTPUT_DIR, RANDOM_STATE
from pipeline.clustering import MIXED_DISTANCE_CONTRACT
from pipeline.data import file_sha256

MANIFEST_FILENAME = "mvdec_experiment_manifest.json"
MANIFEST_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class ExperimentContext:
    """Validated artifact settings and isolated result directory."""

    dataset: str
    n_clusters: int
    output_dir: Path

    def result_path(self, default_path: Path) -> Path:
        """Return one standard result filename inside this context directory."""

        return self.output_dir / default_path.name


def artifact_n_clusters(artifact: dict) -> int:
    """Return a positive cluster count consistent across artifact metadata."""

    if "n_clusters" not in artifact:
        msg = "MvDEC artifact is missing n_clusters."
        raise ValueError(msg)

    n_clusters = int(artifact["n_clusters"])
    if n_clusters < 2:
        msg = "MvDEC artifact requires n_clusters >= 2."
        raise ValueError(msg)

    config = artifact.get("config")
    if isinstance(config, dict) and config.get("n_clusters") is not None:
        if int(config["n_clusters"]) != n_clusters:
            msg = "MvDEC artifact n_clusters does not match config.n_clusters."
            raise ValueError(msg)
    return n_clusters


def _dataset_directory_name(dataset: str) -> str:
    normalized = dataset.strip().lower().replace(" ", "_")
    safe_characters = normalized.replace("_", "").replace("-", "")
    if not normalized or not safe_characters.isalnum():
        msg = f"MvDEC artifact has an unsafe dataset name: {dataset!r}."
        raise ValueError(msg)
    return normalized


def _manifest_payload(
    artifact: dict,
    data_path: Path,
    representation_path: Path,
    n_clusters: int,
    random_state: int,
) -> dict[str, int | str]:
    dataset = artifact.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        msg = "MvDEC artifact is missing a dataset name."
        raise ValueError(msg)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "data_sha256": file_sha256(data_path),
        "representation_sha256": file_sha256(representation_path),
        "n_clusters": n_clusters,
        "kprototypes_n_init": KPROTOTYPES_N_INIT,
        "random_state": random_state,
        "distance_contract": MIXED_DISTANCE_CONTRACT,
    }


def _ensure_manifest(output_dir: Path, payload: dict[str, int | str]) -> None:
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = (
                f"Experiment manifest is invalid: {manifest_path}. "
                "Use a new --output-dir or repair the manifest."
            )
            raise ValueError(msg) from exc
        if existing != payload:
            msg = (
                f"Experiment output directory belongs to another dataset or "
                f"artifact: {output_dir}. Use a separate --output-dir."
            )
            raise ValueError(msg)
        return

    existing_results = list(output_dir.glob("*.csv"))
    if existing_results:
        msg = (
            f"Experiment output directory contains CSV files without a manifest: "
            f"{output_dir}. Use a new --output-dir."
        )
        raise ValueError(msg)

    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_experiment_context(
    artifact: dict,
    data_path: Path,
    representation_path: Path,
    requested_output_dir: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> ExperimentContext:
    """Validate artifact settings and prepare an isolated output directory."""

    n_clusters = artifact_n_clusters(artifact)
    dataset = artifact.get("dataset")
    if not isinstance(dataset, str):
        msg = "MvDEC artifact is missing a dataset name."
        raise ValueError(msg)

    output_dir = requested_output_dir or OUTPUT_DIR / _dataset_directory_name(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _manifest_payload(
        artifact,
        data_path,
        representation_path,
        n_clusters,
        random_state,
    )
    _ensure_manifest(output_dir, payload)
    return ExperimentContext(
        dataset=dataset,
        n_clusters=n_clusters,
        output_dir=output_dir,
    )
