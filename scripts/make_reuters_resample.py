"""Rebuild REUTERS-10K from the RCV1-v2 token files and draw alternative 10K samples.

Follows the DEC/IDEC ``make_reuters_data`` recipe that produced the REUTERS-10K arrays
reused by DEKM: documents whose RCV1-v2 topics contain exactly one of CCAT, GCAT, MCAT
and ECAT, a 2,000-word count vocabulary, sublinear L2 TF-IDF, the first 10,000
documents in file order, scaled by sqrt(2000) and permuted with seed 1234. The rebuilt
canonical sample is compared with the DEKM release arrays, then each ``--sample-seed``
draws 10,000 documents uniformly from the whole single-label corpus instead.

Controlled diagnostic only: it tests whether Table 2 could come from a different
REUTERS-10K sample, which the MvDEC paper does not disclose.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# The JMLR pages now return 404; the Wayback Machine keeps the original files.
MIRRORS = (
    "https://web.archive.org/web/2015id_/http://jmlr.csail.mit.edu/papers/volume5/lewis04a",
    "https://www.jmlr.org/papers/volume5/lewis04a",
)
QRELS = "a08-topic-qrels/rcv1-v2.topics.qrels.gz"
TOKEN_FILES = tuple(
    f"a12-token-files/lyrl2004_tokens_{part}.dat.gz"
    for part in ("test_pt0", "test_pt1", "test_pt2", "test_pt3", "train")
)
CATEGORY_IDS = {"CCAT": 0, "GCAT": 1, "MCAT": 2, "ECAT": 3}
N_FEATURES = 2000
N_SAMPLES = 10000
CANONICAL_PERMUTATION_SEED = 1234
DOWNLOAD_ATTEMPTS = 4


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for relative in (QRELS, *TOKEN_FILES):
        target = raw_dir / Path(relative).name
        if target.exists():
            continue
        if not _fetch_any(relative, target):
            raise RuntimeError(f"Could not download {relative} from any mirror.")


def _fetch_any(relative: str, target: Path) -> bool:
    # Wayback rate-limits bursts (HTTP 429), so retry each mirror with backoff.
    for attempt in range(DOWNLOAD_ATTEMPTS):
        for mirror in MIRRORS:
            try:
                with urllib.request.urlopen(f"{mirror}/{relative}", timeout=300) as r:
                    with target.open("wb") as handle:
                        shutil.copyfileobj(r, handle)
                return True
            except OSError as error:
                print(f"download failed: {mirror}/{relative}: {error}", flush=True)
                target.unlink(missing_ok=True)
        time.sleep(60 * (attempt + 1))
    return False


def single_label_documents(qrels_lines: Iterable[str]) -> dict[int, int]:
    """Map document id to class for documents with exactly one root category."""

    categories: dict[int, list[str]] = {}
    for line in qrels_lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] in CATEGORY_IDS:
            categories.setdefault(int(fields[1]), []).append(fields[0])
    return {
        did: CATEGORY_IDS[cats[0]] for did, cats in categories.items() if len(cats) == 1
    }


def iter_documents(lines: Iterable[str]) -> Iterator[tuple[int, str]]:
    """Yield (document id, token text) from the LYRL2004 ``.I``/``.W`` format."""

    did, doc = None, ""
    for line in lines:
        if line.startswith(".I"):
            if did is not None:
                yield did, doc
            did, doc = int(line.split()[1]), ""
        elif not line.startswith(".W"):
            doc += line
    if did is not None:
        yield did, doc


def _gzip_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from handle


def build_corpus(raw_dir: Path) -> tuple[list[str], np.ndarray]:
    with gzip.open(raw_dir / Path(QRELS).name, "rt", encoding="utf-8") as handle:
        labels = single_label_documents(handle)
    docs, targets = [], []
    token_paths = [raw_dir / Path(name).name for name in TOKEN_FILES]
    for did, text in iter_documents(_gzip_lines(token_paths)):
        if did in labels:
            docs.append(text)
            targets.append(labels[did])
    return docs, np.asarray(targets, dtype=np.int32)


def tfidf_features(docs: list[str]):
    counts = CountVectorizer(dtype=np.float64, max_features=N_FEATURES).fit_transform(
        docs
    )
    return TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(counts)


def dense_sample(tfidf, targets: np.ndarray, rows: np.ndarray):
    x = np.asarray(tfidf[rows].todense(), dtype=np.float32) * np.sqrt(N_FEATURES)
    return x.astype(np.float32), targets[rows]


def save_release_arrays(root: Path, x: np.ndarray, y: np.ndarray) -> None:
    target = root / "REUTERS"
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "10k_feature.npy", x)
    np.save(target / "10k_target.npy", y)


def compare_with_release(x: np.ndarray, y: np.ndarray, release_dir: Path) -> dict:
    x_ref = np.load(release_dir / "10k_feature.npy").astype(np.float32)
    y_ref = np.load(release_dir / "10k_target.npy").reshape(-1)
    return {
        "same_shape": list(x.shape) == list(x_ref.shape),
        "max_abs_feature_diff": float(np.max(np.abs(x - x_ref)))
        if x.shape == x_ref.shape
        else None,
        "label_agreement": (
            float(np.mean(y == y_ref)) if y.shape == y_ref.shape else None
        ),
        "label_counts": np.bincount(y, minlength=4).tolist(),
        "release_label_counts": np.bincount(y_ref, minlength=4).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, action="append", default=[])
    args = parser.parse_args()

    download(args.raw_dir)
    docs, targets = build_corpus(args.raw_dir)
    tfidf = tfidf_features(docs)
    summary: dict = {"n_single_label_documents": len(docs)}

    permutation = np.random.RandomState(CANONICAL_PERMUTATION_SEED).permutation(
        N_SAMPLES
    )
    x, y = dense_sample(tfidf, targets, np.arange(N_SAMPLES)[permutation])
    save_release_arrays(args.output_root / "canonical", x, y)
    summary["canonical_vs_release"] = compare_with_release(x, y, args.release_dir)

    summary["samples"] = {}
    for seed in args.sample_seed:
        rows = np.random.RandomState(seed).choice(len(docs), N_SAMPLES, replace=False)
        x, y = dense_sample(tfidf, targets, rows)
        save_release_arrays(args.output_root / f"sample_{seed}", x, y)
        counts = np.bincount(y, minlength=4).tolist()
        summary["samples"][seed] = {"label_counts": counts}

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
