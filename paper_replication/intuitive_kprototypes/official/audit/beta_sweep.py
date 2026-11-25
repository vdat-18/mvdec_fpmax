"""50 runs per beta of the official procedure, through our equivalent model.

Usage: beta_sweep.py <dataset> <random|select> <beta> [<beta> ...]
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import equivalence as eq  # noqa: E402
from Code import cluster_accuracy  # noqa: E402

key, init = sys.argv[1], sys.argv[2]
data, num, cat, k = eq.load(key)
frame = eq.pd.read_csv(eq.DATA / eq.MANIFEST[key]["processed_csv"].replace("\\", "/"))
truth = eq.pd.factorize(frame["target"].astype(str), sort=True)[0]
p = eq.PARAMS[key]
for beta in [float(b) for b in sys.argv[3:]]:
    scores, failed = [], 0
    for seed in range(50):
        original = eq.ours_module.select_initial_prototypes
        if init == "random":
            eq.ours_module.select_initial_prototypes = lambda *a, s=seed, **kw: (
                eq.official_random_indices(data, k, s)
            )
        try:
            labels = eq.ours_module.IntuitiveKPrototypes(
                k,
                lambda_init=1.0,
                mu_param=p["mu"],
                gamma=p["gamma"],
                beta=beta,
                max_iter=100,
                random_state=seed,
                init_strategy="first",
                mu_convention="swapped",
                strict_init=False,
            ).fit_predict(data[num].to_numpy(float), data[cat].to_numpy(object))
            scores.append(
                (
                    cluster_accuracy(truth, labels),
                    adjusted_rand_score(truth, labels),
                    normalized_mutual_info_score(truth, labels),
                )
            )
        except ValueError:
            failed += 1
        finally:
            eq.ours_module.select_initial_prototypes = original
    s = np.array(scores)
    print(
        f"{key} {init} beta={beta:g} mu={p['mu']} gamma={p['gamma']} runs={len(s)} "
        f"emptyclusters={failed} AC={s[:, 0].mean():.3f}({s[:, 0].std():.3f}) "
        f"ARI={s[:, 1].mean():.3f}({s[:, 1].std():.3f}) "
        f"NMI={s[:, 2].mean():.3f}({s[:, 2].std():.3f})",
        flush=True,
    )
