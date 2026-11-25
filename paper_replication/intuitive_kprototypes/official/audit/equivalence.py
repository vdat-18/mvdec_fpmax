"""Official Code.py vs our model on the paper datasets, same init, same parameters."""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(
    0, str(ROOT / "paper_replication" / "intuitive_kprototypes" / "official")
)
warnings.filterwarnings("ignore")

from Code import IntuitiveKPrototypes as Official  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import intuitive_kprototypes.model as ours_module  # noqa: E402

DATA = ROOT / "paper_replication" / "intuitive_kprototypes" / "data" / "uci"
CONFIG = json.loads((ROOT / "configs" / "fpmax_study.json").read_text())
PARAMS = CONFIG["intuitive"]["per_dataset"]
MANIFEST = {d["key"]: d for d in json.loads((DATA / "manifest.json").read_text())}


def load(key):
    meta = MANIFEST[key]
    frame = pd.read_csv(DATA / meta["processed_csv"].replace("\\", "/"))
    num, cat = meta["numeric_cols"], meta["categorical_cols"]
    data = frame[num + cat].copy()
    for col in cat:
        data[col] = pd.factorize(data[col].astype(str), sort=True)[0]
    return data, num, cat, meta["n_clusters"]


def official_random_indices(data, k, seed):
    return data.reset_index(drop=True).sample(n=k, random_state=seed).index.to_numpy()


def run(key, seeds, init):
    data, num, cat, k = load(key)
    p = PARAMS[key]
    for seed in seeds:
        start = time.perf_counter()
        off = Official(
            k,
            lamda=p["mu"],
            gamma=p["gamma"],
            beta=p["beta"],
            max_iter=100,
            init_prototype=(init == "select"),
            random_state=seed,
        ).fit_predict(data, num, cat)
        off_seconds = time.perf_counter() - start
        original = ours_module.select_initial_prototypes
        if init == "random":
            ours_module.select_initial_prototypes = lambda *a, s=seed, **kw: (
                official_random_indices(data, k, s)
            )
        try:
            model = ours_module.IntuitiveKPrototypes(
                k,
                lambda_init=1.0,
                mu_param=p["mu"],
                gamma=p["gamma"],
                beta=p["beta"],
                max_iter=100,
                random_state=seed,
                init_strategy="first",
                mu_convention="swapped",
                strict_init=False,
            )
            mine = model.fit_predict(
                data[num].to_numpy(float), data[cat].to_numpy(object)
            )
            note = ""
        except Exception as error:  # report, do not hide
            mine, note = None, f"ours raised {type(error).__name__}: {error}"
        finally:
            ours_module.select_initial_prototypes = original
        agree = adjusted_rand_score(off, mine) if mine is not None else float("nan")
        sizes = np.bincount(off, minlength=k).tolist()
        print(
            f"{key:28s} {init:6s} seed={seed} agreeARI={agree:.3f} "
            f"official_sizes={sizes} official_s={off_seconds:.1f} {note}",
            flush=True,
        )


if __name__ == "__main__":
    key, init, *seeds = sys.argv[1:]
    run(key, [int(s) for s in seeds], init)
