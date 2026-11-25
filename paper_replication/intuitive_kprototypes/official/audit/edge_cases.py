"""Edge-case probes of the authors' Code.py."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from Code import IntuitiveKPrototypes, cluster_accuracy  # noqa: E402

# 1. F1 without label matching: swapping the two cluster ids changes F1, not AC.
y = np.array([0] * 6 + [1] * 4)
pred = np.array([0] * 5 + [1] + [1] * 4)
swapped = 1 - pred
print(
    "F1 macro  pred / swapped ids:",
    f1_score(y, pred, average="macro"),
    f1_score(y, swapped, average="macro"),
)
print(
    "AC        pred / swapped ids:",
    cluster_accuracy(y, pred),
    cluster_accuracy(y, swapped),
)

# 2. NMI normalisation: sklearn default (arithmetic) vs paper Eq. (37) (geometric).
rng = np.random.default_rng(0)
truth = rng.integers(0, 5, 303)
noisy = np.where(rng.random(303) < 0.5, truth, rng.integers(0, 3, 303))
print(
    "NMI arithmetic / geometric:",
    normalized_mutual_info_score(truth, noisy),
    normalized_mutual_info_score(truth, noisy, average_method="geometric"),
)

# 3. Eq. (12) d* = 0 branch: a numeric attribute constant inside both clusters.
model = IntuitiveKPrototypes(2, lamda=0.3, gamma=0.2, beta=2, random_state=0)
df = pd.DataFrame({"a": [0.0] * 5 + [1.0] * 5, "b": np.linspace(0, 1, 10)})
labels = np.array([0] * 5 + [1] * 5)
centers, _, _ = model._update_intuitionistic_centroid(df, labels, ["a", "b"], [])
print(
    "S_n with a constant-per-cluster attribute:",
    model._num_inter_similarity(df, labels, ["a", "b"], centers).to_dict(),
)
df2 = pd.DataFrame({"a": [0.5] * 10, "b": np.linspace(0, 1, 10)})
centers2, _, _ = model._update_intuitionistic_centroid(df2, labels, ["a", "b"], [])
s2 = model._num_inter_similarity(df2, labels, ["a", "b"], centers2)
print("S_n with a globally constant attribute (paper: 1):", s2.to_dict())
xi2 = model._num_intra_complexity(df2, labels, ["a", "b"], centers2)
print("weights then:", model._calc_num_weight(xi2, s2).to_dict())

# 4. Empty cluster: what the official loop does with NaN centres.
d_num = np.array([np.nan, 0.4])
print("np.argmin([nan, 0.4]) ->", int(np.argmin(d_num)), "(NaN wins argmin)")
