import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

EPS = 1e-9


def cluster_accuracy(y_true, y_pred):
    """
    Clustering Accuracy with Hungarian algorithm for label permutation.
    Fix the bug that directly using accuracy_score on clustering labels is wrong.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    n_true = len(true_labels)
    n_pred = len(pred_labels)
    cost_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            mask_true = y_true == true_labels[i]
            mask_pred = y_pred == pred_labels[j]
            cost_matrix[i, j] = np.sum(np.logical_and(mask_true, mask_pred))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    total_correct = cost_matrix[row_ind, col_ind].sum()
    return total_correct / len(y_true)


def evaluate_clustering(y_true, y_pred):
    """
    Evaluate clustering performance using standard metrics in the paper.
    Metrics: Accuracy(AC), Macro-F1, Adjusted Rand Index(ARI), Normalized Mutual Information(NMI).
    """
    acc = cluster_accuracy(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    res = {"AC": acc, "F1(macro)": f1, "ARI": ari, "NMI": nmi}
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
    return res


class IntuitiveKPrototypes:
    """
    Intuitive‑K‑prototypes: A mixed data clustering algorithm with intuitionistic distribution centroid
    Paper: Intuitive‑K‑prototypes: A mixed data clustering algorithm with intuitionistic distribution centroid, Pattern Recognition 158 (2025)
    Reference: Wang H, Mi J. Intuitive‑K‑prototypes: A mixed data clustering algorithm with intuitionistic distribution centroid[J]. Pattern Recognition, 2025, 158: 111062.
    """
    def __init__(self,
                 n_clusters: int,
                 lamda: float = 0.2,
                 gamma: float = 0.5,
                 beta: float = 2.0,
                 belt: float = 1.0,
                 max_iter: int = 100,
                 init_prototype: bool = True,
                 verbose: bool = False,
                 random_state: int = 42):
        # Parameter validation
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if not (0 <= lamda <= 1):
            raise ValueError("lamda must be in [0, 1]")
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")
        if beta <= 1.0:
            raise ValueError("beta must be > 1, recommended value is 2")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")

        self.n_clusters = n_clusters
        self.lamda = lamda
        self.gamma = gamma
        self.beta = beta
        self.belt = belt
        self.max_iter = max_iter
        self.init_prototype = init_prototype
        self.verbose = verbose
        self.random_state = random_state

        # Model output attributes
        self.labels_ = None
        self.numerical_centers = None
        self.membership_centers = None
        self.non_membership_centers = None
        self.num_weights = None
        self.cat_weights = None
        self.cost_history = []

    @staticmethod
    def _mixed_distance(x_num, x_cat, c_num, c_cat):
        """
        Eq.(1): Simple mixed-type distance for initialization.
        Numerical: Manhattan distance; Categorical: Hamming distance.
        """
        d_num = np.sum(np.abs(x_num - c_num))
        d_cat = np.sum(x_cat != c_cat)
        return d_num + d_cat

    def _select_initial_prototypes(self, df: pd.DataFrame, num_cols, cat_cols):
        n = df.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx0 = rng.integers(0, n)
        x0 = df.iloc[idx0]

        # 找 x1
        dist0 = [self._mixed_distance(df.iloc[i][num_cols], df.iloc[i][cat_cols], x0[num_cols], x0[cat_cols]) for i in
                 range(n)]
        idx1 = int(np.argmax(dist0))
        x1 = df.iloc[idx1]

        # 找 x2
        dist1 = [self._mixed_distance(df.iloc[i][num_cols], df.iloc[i][cat_cols], x1[num_cols], x1[cat_cols]) for i in
                 range(n)]
        idx2 = int(np.argmax(dist1))
        x2 = df.iloc[idx2]

        d_approx = self._mixed_distance(x1[num_cols], x1[cat_cols], x2[num_cols], x2[cat_cols])
        d_lambda = self.belt * d_approx / self.n_clusters

        selected_indices = [idx1, idx2]

        # 寻找剩余原型
        while len(selected_indices) < self.n_clusters:
            found = False
            for i in range(n):
                if i in selected_indices:
                    continue
                xi = df.iloc[i]

                dists_q = []
                for idx in selected_indices:
                    q_row = df.iloc[idx]
                    d = self._mixed_distance(xi[num_cols], xi[cat_cols], q_row[num_cols], q_row[cat_cols])
                    dists_q.append(d)

                if all(d >= d_lambda for d in dists_q):
                    selected_indices.append(i)
                    found = True
                    break

            if not found:
                # 如果找不到满足条件的点，降低阈值重新寻找
                d_lambda *= 0.8
                # 如果阈值降到极小，强制选取距离现有原型最远的点
                if d_lambda < 1e-6:
                    best_i = -1
                    max_min_dist = -1
                    for i in range(n):
                        if i in selected_indices:
                            continue
                        xi = df.iloc[i]
                        dists_q = []
                        for idx in selected_indices:
                            q_row = df.iloc[idx]
                            d = self._mixed_distance(xi[num_cols], xi[cat_cols], q_row[num_cols], q_row[cat_cols])
                            dists_q.append(d)
                        min_dist = min(dists_q)
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                            best_i = i
                    if best_i != -1:
                        selected_indices.append(best_i)
                    else:
                        break  # 无法再选出新点

        return df.iloc[selected_indices].reset_index(drop=True)

    def _update_intuitionistic_centroid(self, df, labels, num_cols, cat_cols):
        """
        Definition 1: Update intuitionistic distribution centroid, Eq.(2)(3)(4).
        """
        k = self.n_clusters
        num_centers_list = []
        mem_centers = {cid: {} for cid in range(k)}
        nonmem_centers = {cid: {} for cid in range(k)}

        for cid in range(k):
            sub = df[labels == cid]
            # Numerical centroid: mean value
            if len(num_cols) > 0:
                nc = sub[num_cols].mean()
            else:
                nc = pd.Series(dtype=np.float64)
            num_centers_list.append(nc)

            # Categorical centroid: intuitionistic distribution
            for col in cat_cols:
                all_val_counts = df[col].value_counts().to_dict()
                sub_val_counts = sub[col].value_counts().to_dict()
                mu_dict = {}
                nu_dict = {}
                for val, total_cnt in all_val_counts.items():
                    in_cluster = sub_val_counts.get(val, 0)
                    size_cluster = len(sub)
                    # Eq.(3) membership degree
                    mu = (in_cluster ** 2) / (total_cnt * size_cluster) if (total_cnt * size_cluster) > 0 else 0
                    # Eq.(4) non-membership degree
                    sum_s = 0.0
                    for s in range(k):
                        if s == cid:
                            continue
                        cnt_s = np.sum(df.loc[labels == s, col] == val)
                        sum_s += (cnt_s / total_cnt) ** 2 if total_cnt > 0 else 0
                    nu = sum_s
                    mu_dict[val] = mu
                    nu_dict[val] = nu
                mem_centers[cid][col] = mu_dict
                nonmem_centers[cid][col] = nu_dict

        # Explicitly bind column names to prevent KeyError
        num_centers = pd.DataFrame(num_centers_list, columns=num_cols)
        return num_centers, mem_centers, nonmem_centers

    def _cat_intuitionistic_entropy(self, labels, mem_centers, nonmem_centers, cat_cols, n):
        """Definition 5: Intra-cluster complexity for categorical attributes, Eq.(17)(18)."""
        k = self.n_clusters
        cluster_size = np.bincount(labels, minlength=k)
        E_c = pd.Series(index=cat_cols, dtype=np.float64)
        for col in cat_cols:
            total_entropy = 0.0
            for cid in range(k):
                mu_d = mem_centers[cid][col]
                nu_d = nonmem_centers[cid][col]
                sum_min = 0.0
                sum_max = 0.0
                for v in mu_d.keys():
                    mu = mu_d[v]
                    nu = nu_d[v]
                    sum_min += min(mu, nu)
                    sum_max += max(mu, nu)
                ent = sum_min / sum_max if sum_max > EPS else 0
                alpha_i = cluster_size[cid] / n
                total_entropy += alpha_i * ent
            E_c[col] = total_entropy
        return E_c

    def _cat_inter_similarity(self, labels, mem_centers, nonmem_centers, cat_cols, n):
        """Definition 6: Inter-cluster similarity for categorical attributes, Eq.(19)(20)."""
        k = self.n_clusters
        cluster_size = np.bincount(labels, minlength=k)
        S_c = pd.Series(index=cat_cols, dtype=np.float64)
        for col in cat_cols:
            total_sim = 0.0
            D_sum = 0
            for i in range(k):
                for j in range(i + 1, k):
                    ni = cluster_size[i]
                    nj = cluster_size[j]
                    weight_ij = ni + nj
                    D_sum += weight_ij
                    mu_i = mem_centers[i][col]
                    nu_i = nonmem_centers[i][col]
                    mu_j = mem_centers[j][col]
                    nu_j = nonmem_centers[j][col]
                    values = list(mu_i.keys())
                    t = len(values)
                    abs_sum = 0.0
                    for v in values:
                        diff_i = mu_i[v] - nu_i[v]
                        diff_j = mu_j[v] - nu_j[v]
                        abs_sum += abs(diff_i - diff_j)
                    sim_ij = 1.0 - (abs_sum / (2 * t)) if t > 0 else 1
                    total_sim += weight_ij * sim_ij
            S_c[col] = total_sim / D_sum if D_sum > EPS else 1.0
        return S_c

    def _calc_cat_weight(self, E_c: pd.Series, S_c: pd.Series):
        """Definition 7: Categorical attribute weights, Eq.(21)(22)."""
        phi_c = self.gamma * E_c + (1 - self.gamma) * S_c
        synth = phi_c ** (1 / (self.beta - 1))
        inv_synth = np.where(np.isinf(1 / synth), 0, 1 / synth)
        denom = np.sum(inv_synth)
        raw_w = inv_synth / denom if denom > EPS else np.ones_like(inv_synth) / len(inv_synth)
        w_new = raw_w * len(E_c)
        # 修复：重新封装为Series，保留列名索引
        return pd.Series(w_new, index=E_c.index)
    def _num_intra_complexity(self, df, labels, num_cols, num_centers):
        """Definition 2: Intra-cluster complexity for numerical attributes, Eq.(5)."""
        k = self.n_clusters
        n = len(df)
        cluster_size = np.bincount(labels, minlength=k)
        xi_cluster = []
        for cid in range(k):
            sub = df.loc[labels == cid, num_cols]
            mu_c = num_centers.loc[cid, num_cols]
            std_c = sub.std(ddof=0)
            xi = np.where(np.abs(mu_c) < EPS, 0, std_c / mu_c)
            xi_cluster.append(xi)
        xi_df = pd.DataFrame(xi_cluster, columns=num_cols)
        xi_max = xi_df.max(axis=0)
        xi_norm = xi_df / xi_max.replace(0, EPS)
        xi_n = pd.Series(dtype=np.float64, index=num_cols)
        for col in num_cols:
            s = 0.0
            for cid in range(k):
                alpha_i = cluster_size[cid] / n
                s += alpha_i * xi_norm.loc[cid, col]
            xi_n[col] = s
        return xi_n

    def _num_inter_similarity(self, df, labels, num_cols, num_centers):
        """Inter-cluster similarity for numerical attributes, granular-ball global-local radius."""
        k = self.n_clusters
        cluster_size = np.bincount(labels, minlength=k)
        n = len(df)
        S_n = pd.Series(index=num_cols, dtype=np.float64)
        for col in num_cols:
            total = 0.0
            D_sum = 0
            for i in range(k):
                sub_i = df.loc[labels == i, col]
                mi = num_centers.loc[i, col]
                dist_i = np.abs(sub_i - mi)
                r_global_i = dist_i.mean()
                mask_i = dist_i <= r_global_i
                good_i = sub_i[mask_i]
                r_local_i = np.mean(np.abs(good_i - mi)) if len(good_i) > 0 else 0.0

                for j in range(i + 1, k):
                    nj = cluster_size[j]
                    ni = cluster_size[i]
                    weight_ij = ni + nj
                    D_sum += weight_ij
                    sub_j = df.loc[labels == j, col]
                    mj = num_centers.loc[j, col]
                    dist_j = np.abs(sub_j - mj)
                    r_global_j = dist_j.mean()
                    mask_j = dist_j <= r_global_j
                    good_j = sub_j[mask_j]
                    r_local_j = np.mean(np.abs(good_j - mj)) if len(good_j) > 0 else 0.0

                    d_ji_global = np.mean(np.abs(sub_j - mi))
                    d_ij_global = np.mean(np.abs(sub_i - mj))
                    d_ji_local = np.mean(np.abs(good_j - mi)) if len(good_j) > 0 else EPS
                    d_ij_local = np.mean(np.abs(good_i - mj)) if len(good_i) > 0 else EPS

                    denom1 = d_ji_global * d_ji_local
                    denom2 = d_ij_global * d_ij_local
                    h = (r_global_i * r_local_i / denom1) + (r_global_j * r_local_j / denom2)
                    sim_att = 1 - np.exp(-h)
                    total += weight_ij * sim_att
            S_n[col] = total / D_sum if D_sum > EPS else 1.0
        return S_n

    def _calc_num_weight(self, xi_n: pd.Series, S_n: pd.Series):
        """Definition 4: Numerical attribute weights, Eq.(15)(16)."""
        phi_n = self.gamma * xi_n + (1 - self.gamma) * S_n
        synth = phi_n ** (1 / (self.beta - 1))
        inv_synth = np.where(np.isinf(1 / synth), 0, 1 / synth)
        denom = np.sum(inv_synth)
        raw_w = inv_synth / denom if denom > EPS else np.ones_like(inv_synth) / len(inv_synth)
        w_new = raw_w * len(xi_n)
        # 修复：重新封装为Series，保留列名索引
        return pd.Series(w_new, index=xi_n.index)

    def _cat_composite_distance_sample(self, x_cat_row, mem_centers, nonmem_centers, cat_weights, cat_cols):
        """Definition 8 & 9: Membership + non-membership composite distance, Eq.(24-26)."""
        k = self.n_clusters
        dist_all = []
        for cid in range(k):
            d1_total = 0.0
            d2_total = 0.0
            for col in cat_cols:
                xv = x_cat_row[col]
                mu_dict = mem_centers[cid][col]
                nu_dict = nonmem_centers[cid][col]
                d1 = 0.0
                d2 = 0.0
                for val in mu_dict.keys():
                    mu = mu_dict[val]
                    nu = nu_dict[val]
                    if xv != val:
                        d1 += mu
                    if xv == val:
                        d2 += nu
                d1_total += cat_weights[col] * d1
                d2_total += cat_weights[col] * d2
            composite = (1 - self.lamda) * d1_total + self.lamda * d2_total
            dist_all.append(composite)
        return np.array(dist_all)

    def _num_weighted_distance_sample(self, x_num_row, num_centers, num_weights, num_cols):
        """Weighted Manhattan distance for numerical attributes."""
        dists = []
        for cid in range(self.n_clusters):
            center_row = num_centers.loc[cid, num_cols]
            d = np.sum(num_weights * np.abs(x_num_row - center_row))
            dists.append(d)
        return dists

    def _compute_cost(self, df, labels, num_centers, mem_centers, nonmem_centers, num_weights, cat_weights, num_cols, cat_cols):
        """Eq.(27): Total clustering cost function for convergence monitoring."""
        total_cost = 0.0
        n = len(df)
        for i in range(n):
            cid = labels[i]
            d_n = 0.0
            if len(num_cols) > 0:
                xn = df.loc[i, num_cols]
                d_n = np.sum(num_weights * np.abs(xn - num_centers.loc[cid, num_cols]))
            d_c = 0.0
            if len(cat_cols) > 0:
                xc = df.loc[i, cat_cols]
                d1_total = 0.0
                d2_total = 0.0
                for col in cat_cols:
                    xv = xc[col]
                    mu_dict = mem_centers[cid][col]
                    nu_dict = nonmem_centers[cid][col]
                    d1 = 0.0
                    d2 = 0.0
                    for val in mu_dict.keys():
                        mu = mu_dict[val]
                        nu = nu_dict[val]
                        if xv != val:
                            d1 += mu
                        if xv == val:
                            d2 += nu
                    d1_total += cat_weights[col] * d1
                    d2_total += cat_weights[col] * d2
                d_c = (1 - self.lamda) * d1_total + self.lamda * d2_total
            total_cost += d_n + d_c
        return total_cost

    def fit_predict(self, df: pd.DataFrame, num_cols: list, cat_cols: list):
        """
        Fit model and return predicted cluster labels.
        :param df: Input dataset (without ground-truth label)
        :param num_cols: List of numerical column names
        :param cat_cols: List of categorical column names
        :return: Predicted cluster labels
        """
        # Input validation
        missing_num = [c for c in num_cols if c not in df.columns]
        missing_cat = [c for c in cat_cols if c not in df.columns]
        if missing_num or missing_cat:
            raise KeyError(f"Columns not found in dataframe. Missing num: {missing_num}, Missing cat: {missing_cat}")

        df_work = df.copy().reset_index(drop=True)
        n = df_work.shape[0]
        has_num = len(num_cols) > 0
        has_cat = len(cat_cols) > 0

        # Initialize prototypes
        if self.init_prototype:
            Q = self._select_initial_prototypes(df_work, num_cols, cat_cols)
        else:
            Q = df_work.sample(n=self.n_clusters, random_state=self.random_state, replace=False)

        # Initial assignment (FIXED: argmin instead of argmax)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            xi_num = df_work.loc[i, num_cols] if has_num else []
            xi_cat = df_work.loc[i, cat_cols] if has_cat else []
            d_list = []
            for _, q_row in Q.iterrows():
                d = self._mixed_distance(xi_num, xi_cat, q_row[num_cols], q_row[cat_cols])
                d_list.append(d)
            labels[i] = int(np.argmin(d_list)) if len(d_list) > 0 else 0

        iter_cnt = 0
        self.cost_history = []

        # Main iterative loop
        while iter_cnt < self.max_iter:
            old_labels = labels.copy()

            # Empty cluster detection
            cluster_counts = np.bincount(labels, minlength=self.n_clusters)
            if np.any(cluster_counts == 0):
                warnings.warn(f"Empty cluster detected at iteration {iter_cnt}, result may be unstable.")

            # Step 1: update centroids
            num_centers, mem_centers, nonmem_centers = self._update_intuitionistic_centroid(
                df_work, labels, num_cols, cat_cols
            )

            # Step 2: compute attribute weights
            if has_cat:
                E_c = self._cat_intuitionistic_entropy(labels, mem_centers, nonmem_centers, cat_cols, n)
                S_c = self._cat_inter_similarity(labels, mem_centers, nonmem_centers, cat_cols, n)
                cat_weights = self._calc_cat_weight(E_c, S_c)
            else:
                cat_weights = np.array([])

            if has_num:
                xi_n = self._num_intra_complexity(df_work, labels, num_cols, num_centers)
                S_n = self._num_inter_similarity(df_work, labels, num_cols, num_centers)
                num_weights = self._calc_num_weight(xi_n, S_n)
            else:
                num_weights = np.array([])

            # Step 3: reassign samples
            new_labels = np.zeros(n, dtype=int)
            for i in range(n):
                d_n = np.zeros(self.n_clusters)
                if has_num:
                    xn = df_work.loc[i, num_cols]
                    d_n = self._num_weighted_distance_sample(xn, num_centers, num_weights, num_cols)

                d_c = np.zeros(self.n_clusters)
                if has_cat:
                    xc = df_work.loc[i, cat_cols]
                    d_c = self._cat_composite_distance_sample(xc, mem_centers, nonmem_centers, cat_weights, cat_cols)

                total_dist = d_n + d_c
                new_labels[i] = int(np.argmin(total_dist))

            labels = new_labels
            iter_cnt += 1

            # Record cost
            if self.verbose:
                cost = self._compute_cost(df_work, labels, num_centers, mem_centers, nonmem_centers,
                                          num_weights, cat_weights, num_cols, cat_cols)
                self.cost_history.append(cost)
                print(f"Iter {iter_cnt:3d} | Cost: {cost:.4f} | Changed: {np.sum(labels != old_labels)} samples")

            # Early stop
            if np.array_equal(labels, old_labels):
                if self.verbose:
                    print(f"Converged at iteration {iter_cnt}")
                break

        # Save model state
        self.labels_ = labels
        self.numerical_centers = num_centers if has_num else None
        self.membership_centers = mem_centers if has_cat else None
        self.non_membership_centers = nonmem_centers if has_cat else None
        self.num_weights = num_weights if has_num else None
        self.cat_weights = cat_weights if has_cat else None

        return labels

    def predict(self, df_new: pd.DataFrame, num_cols: list, cat_cols: list):
        """Predict cluster labels for new unseen samples."""
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet. Call fit_predict() first.")

        df_work = df_new.copy().reset_index(drop=True)
        n = len(df_work)
        has_num = len(num_cols) > 0
        has_cat = len(cat_cols) > 0
        pred_labels = np.zeros(n, dtype=int)

        for i in range(n):
            d_n = np.zeros(self.n_clusters)
            if has_num:
                xn = df_work.loc[i, num_cols]
                d_n = self._num_weighted_distance_sample(xn, self.numerical_centers, self.num_weights, num_cols)

            d_c = np.zeros(self.n_clusters)
            if has_cat:
                xc = df_work.loc[i, cat_cols]
                d_c = self._cat_composite_distance_sample(xc, self.membership_centers,
                                                          self.non_membership_centers, self.cat_weights, cat_cols)

            total_dist = d_n + d_c
            pred_labels[i] = int(np.argmin(total_dist))
        return pred_labels


if __name__ == "__main__":
    # ========= Cleveland Heart Disease Dataset (UCI) =========
    # 13 attributes + 1 label, total 14 columns
    features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'label'
    ]
    # Numerical attributes (5 items, consistent with paper)
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # Categorical attributes (8 items, consistent with paper)
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    fp = "processed.cleveland.txt"
    n_clusters = 2  # set to 5 for 5-cluster experiment
    binary_label = True  # True: binary (0 vs 1-4); False: 5-class (0,1,2,3,4)

    if os.path.exists(fp):
        # Robust data loading for UCI Cleveland dataset
        df_raw = pd.read_csv(
            fp,
            header=None,
            sep=",",
            na_values="?",
            engine="python"
        )

        # Verify column count
        if df_raw.shape[1] != len(features):
            raise ValueError(
                f"Column count mismatch: file has {df_raw.shape[1]} columns, "
                f"but {len(features)} feature names provided."
            )
        df_raw.columns = features

        # Fill missing values (consistent with paper: mode for categorical, median for numerical)
        for col in cat_features:
            df_raw[col].fillna(df_raw[col].mode()[0], inplace=True)
        for col in num_features:
            df_raw[col].fillna(df_raw[col].median(), inplace=True)

        # Process ground truth labels
        y_gt = df_raw["label"].values
        if binary_label:
            # Binarize: 0 = no disease, 1 = disease (original 1/2/3/4 are all positive)
            y_gt = (y_gt > 0).astype(int)

        df_data = df_raw[num_features + cat_features].copy()

        # Encode all categorical features (convert to string first to avoid type issues)
        encoders = {}
        for col in cat_features:
            le = LabelEncoder()
            df_data[col] = le.fit_transform(df_data[col].astype(str))
            encoders[col] = le

        # Normalize numerical attributes (MinMax scaling, consistent with paper)
        scaler = MinMaxScaler()
        df_data[num_features] = scaler.fit_transform(df_data[num_features])

        # Print dataset info for verification
        print(f"Dataset loaded: {df_data.shape[0]} samples, "
              f"{len(num_features)} numerical features, {len(cat_features)} categorical features")
        print(f"Number of clusters: {n_clusters}")
        print(f"Label distribution: {dict(pd.Series(y_gt).value_counts().sort_index())}\n")

        # Run clustering (parameters reference paper experimental settings)
        model = IntuitiveKPrototypes(
            n_clusters=n_clusters,
            lamda=0.3,
            gamma=0.2,
            beta=6,
            max_iter=100,
            init_prototype=False,
            verbose=True,
            random_state=9
        )
        y_pred = model.fit_predict(df_data, num_cols=num_features, cat_cols=cat_features)

        print("\n=== Clustering Performance ===")
        evaluate_clustering(y_gt, y_pred)

