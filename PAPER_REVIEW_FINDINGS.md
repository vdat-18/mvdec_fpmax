# Review toàn diện: mvdec_fpmax (đối chiếu DEKM 2021 / MvDEC 2025)

Ngày review: 2026-07-20. Phạm vi: toàn bộ pipeline so sánh K-means / DEKM / MvDEC /
MiMvDEC without FFS / MiMvDEC with FFS, đối chiếu `docs/2021_Deep Embedded K-Means
Clustering.pdf` và `docs/2025_Multi-view Deep Embedded Clustering...pdf`.

Chưa sửa code nào — đây là tài liệu findings để quyết định thứ tự xử lý.

---

## PHẦN 1 — FINDINGS

### 🔴 CRITICAL

#### C1. MvDEC và MiMvDEC được chấm bằng hai cây thước khác nhau — lỗ hổng lớn nhất

Baseline đưa vào FFS là **Gower thuần số** trên `h_fused` với nhãn MvDEC
([cli.py:306](src/cli.py#L306) → [data.py:394-399](src/pipeline/data.py#L394-L399)).
Nhưng ứng viên lại được chấm bằng **Gower hỗn hợp** (số + Jaccard bất đối xứng)
([forward_selection.py:105-108](src/pipeline/forward_selection.py#L105-L108) →
[clustering.py:98-130](src/pipeline/clustering.py#L98-L130)).

Thực nghiệm kiểm chứng — **giữ nguyên y hệt nhãn MvDEC, không gom cụm lại**, chỉ
thêm cột FP-Max vào công thức khoảng cách:

| config | #feat | silhouette | Δ vs baseline 0.5355 |
|---|---|---|---|
| kmeans, bins=3, sup=0.30 | 3 | 0.5915 | **+0.0559** |
| kmeans, bins=3, sup=0.15 | 7 | 0.5544 | +0.0189 |
| kmeans, bins=7, sup=0.15 | 14 | 0.3627 | −0.1729 |
| quantile, bins=3, sup=0.30 | 15 | 0.2501 | −0.2855 |

Điểm số nhảy trong dải −0.29 → +0.06 **mà cụm không hề đổi**. Vì grid lấy `max`
trên 171 cấu hình, một phần "cải thiện" chỉ là grid tìm ra chỗ cây thước co giãn
có lợi nhất.

**Cơ chế**: trong [clustering.py:122-130](src/pipeline/clustering.py#L122-L130),
trọng số phần nhị phân là `binary_union_count` theo cặp. Càng thêm itemset, phần
số càng bị pha loãng. Đo silhouette trên đúng không gian vừa tối ưu = tự chấm bài
mình.

**Điểm cộng**: repo *đã có* cơ chế đúng — `mvdec-evaluate-best` chấm nhãn MvDEC
trên **cùng** biểu diễn hỗn hợp, xuất `silhouette_delta_vs_mvdec`
([best_evaluation.py:378-443](src/pipeline/best_evaluation.py#L378-L443)). Nhưng
Table 6 của bản thảo không dùng con số này.

#### C2. Artifact air-pollution trong repo bị chính loader của repo từ chối

```
ValueError: This artifact uses the legacy full-view-output average.
```

`data/preprocessed_data/airpollution_demvk_fused_representation.pkl` có
`fusion_contract='mvdec2025_figure_output_average'`, `h_fused` shape **(3765, 23)**
(10 latent + 13 input gốc), `final_training_objective='dekm2021_greedy_...'`,
**không có** `preprocessing`/`stop_reason`/`greedy_eigen_direction`. Loader chặn
tại [data.py:324-331](src/pipeline/data.py#L324-L331).

→ Toàn bộ Table 1 (air pollution) **không tái lập được** bằng code hiện tại.

#### C3. "Multi-seed" cho Tiki/Air Pollution là giả — mọi run ghi đè cùng một file

[MVDEC_dense.py:1437](src/representation_learning/MVDEC_dense.py#L1437):
`run_artifact_path = args.artifact_path` — hằng số qua các run. CSV assignments
cũng vậy ([:1243-1245](src/representation_learning/MVDEC_dense.py#L1243-L1245)).
Public datasets thì đúng: tên file có seed
([:1462-1465](src/representation_learning/MVDEC_dense.py#L1462-L1465)).

Bằng chứng: artifact Tiki hiện tại có `config.seed = 44` — run **cuối cùng** của
`--seed 42 --runs 3`, hai run trước bị xoá sổ.

Hệ quả: `mvdec-evaluate-best --seeds 40 41 42 43 44` chỉ đổi seed **của bước gom
cụm**, biểu diễn deep vẫn cố định. `silhouette_std_across_seeds` không đo độ biến
thiên full-pipeline — bỏ qua nguồn nhiễu lớn nhất (khởi tạo autoencoder + pretrain).

#### C4. Lấy max trên 171 cấu hình rồi so với 1 baseline, không holdout, không kiểm định

Grid = 3 strategies × 3 n_bins × 19 supports
([fpmax.py:18-19](src/pipeline/fpmax.py#L18-L19)) = 171, rồi `idxmax`
([best_evaluation.py:98](src/pipeline/best_evaluation.py#L98)). Chồng thêm: max
qua 2 init K-Prototypes ([clustering.py:291](src/pipeline/clustering.py#L291)),
max qua lưới Intuitive (mu×gamma×beta = 324, hoặc ×9 alpha).

Toàn repo: **không có bất kỳ kiểm định thống kê nào** (t-test/Wilcoxon/Friedman/
Nemenyi/CI). Baseline MvDEC là một con số duy nhất, không grid, không max. So
sánh "vô địch của 171 lượt thi" với "thí sinh thi một lần".

#### C5. Artifact Tiki gắn nhãn sai cấu hình, và chạy lại sẽ ghi đè im lặng

Artifact hiện tại: `config.loss_weights.kmeans = 1.0`. Mặc định hiện tại là
`DEFAULT_LAMBDA_KMEANS = 0.0`
([MVDEC_dense.py:43](src/representation_learning/MVDEC_dense.py#L43)) — quyết
định đã chốt 2026-07-16 theo DEKM Fig. 4.

`loss_ablation_tag` trả chuỗi rỗng cho mặc định
([:165-171](src/representation_learning/MVDEC_dense.py#L165-L171)), nên file
λ2=0 sẽ ghi đè lên file λ2=1 **tại cùng đường dẫn không hậu tố, không cảnh báo**.
File đang nằm ở đó là hàng cũ đội lốt hàng mặc định.

---

### 🟠 HIGH

#### H1. DEKM vs MvDEC không công bằng ở 3 siêu tham số cùng lúc

| | DEKM | MvDEC |
|---|---|---|
| K-Means `n_init` | suy giảm `2×n_iter_` (~10–30) — [DEKM_dense.py:103](src/representation_learning/DEKM_dense.py#L103) | cố định **100** — [MVDEC_dense.py:57](src/representation_learning/MVDEC_dense.py#L57) |
| refresh cadence | mỗi **10** batch — [:164](src/representation_learning/DEKM_dense.py#L164) | mỗi **epoch** (~39 batch REUTERS) — [:843](src/representation_learning/MVDEC_dense.py#L843) |
| ngân sách gradient tối đa | 14.000 step — [:95](src/representation_learning/DEKM_dense.py#L95) | 54.600 step — [:844](src/representation_learning/MVDEC_dense.py#L844) |

MvDEC được ~4× số bước cập nhật và ~5× số lần khởi tạo K-Means. README:271-274 nói
đã bỏ hành vi `2*n_iter_` — nhưng chỉ bỏ ở MvDEC, DEKM vẫn giữ.

#### H2. MiMvDEC không chạy nổi trên public datasets vì RAM

`compute_mixed_gower_distance` tạo ma trận n×n float64:
- REUTERS n=10.000 → ~800 MB/ma trận
- 20NEWS n=18.846 → ~2.84 GB/ma trận

View-weighted precompute **một ma trận cho mỗi alpha**, 9 alpha
([experiments.py:83](src/pipeline/experiments.py#L83),
[:3822-3833](src/pipeline/experiments.py#L3822-L3833)) → ~7 GB ở n=10.000. FFS còn
tính lại ma trận cho **mỗi candidate**
([forward_selection.py:105](src/pipeline/forward_selection.py#L105)).

README:224-228 hướng dẫn chạy đúng lệnh đó trên REUTERS.

#### H3. Bản thảo mô tả biểu diễn 11 chiều, code hiện tại 5 chiều — số liệu Tiki không khớp

Draft dòng 300: "11 features: 7 original + 4 latent". `config.py:7` còn giữ
`H_FUSED_COLUMNS` 11 cột (tàn dư). Artifact hiện tại: `h_fused (1799, 5)`
encoder-average.

Đo trên artifact hiện tại: **MvDEC Gower silhouette = 0.5355**, draft ghi
MvDEC = **40.97**. Khoảng cách với MiMvDEC (55.79) co từ +14.8 điểm xuống
+2.2 điểm — thay đổi hoàn toàn câu chuyện của paper.

#### H4. Baselines air-pollution là số chép từ paper khác, không phải tự chạy

Draft Table 1: K-Means 28.29 / DEKM 50.48 / single-view 52.76 — trùng Table 5 của
MvDEC 2025. Multi-view: **draft ghi 61.97, paper gốc ghi 61.76** (lỗi chép số).
Log của chính repo (`log_history/AIRPOLLUTION.csv`) cho **0.6495**.

Table 1 so *proposed method tự chạy, Gower hỗn hợp, scaler Min-Max tự giả định*
với *baselines từ paper khác, Euclid, scaler không công bố*.

Thêm nữa: **MiMvDEC without FFS = 71.04 và với FFS = 71.04, y hệt nhau** → FFS
đóng góp đúng 0 trên air pollution, nhưng vẫn được liệt kê là contribution #2 và
text dòng 114 lờ đi chuyện này.

#### H5. Không có producer cho baseline "MvDEC single-view"

README:528 hướng dẫn `--source "Single-view=output/single_view_seed42.pkl"`,
nhưng không có code nào sinh ra file đó. Baseline này không tái lập được.

#### H6. Ngưỡng chấp nhận FFS = 1e-6, dừng tham lam, không kiểm định

[forward_selection.py:177,269](src/pipeline/forward_selection.py#L177) và
[experiments.py:3526,3606](src/pipeline/experiments.py#L3526). Mọi cải thiện dù
nhiễu số học đều được nhận. Kết hợp C1 + C4 = công thức overfitting metric.

---

### 🟡 MEDIUM

| # | Vấn đề | Vị trí | Hệ quả |
|---|---|---|---|
| M1 | `number_of_batches` làm tròn khác validator (round vs ceil) | [MVDEC_dense.py:228-233](src/representation_learning/MVDEC_dense.py#L228-L233) vs [data.py:158](src/pipeline/data.py#L158) | Bom hẹn giờ khi mở validator cho Tiki |
| M2 | Pretrain weights ghi vào CWD tên cố định | [MVDEC_dense.py:566,623,703,722](src/representation_learning/MVDEC_dense.py#L566) | Hai config chạy song song đạp weights của nhau |
| M3 | K-Means baseline chạy cả k-means++ và random, aggregate tách theo init | [run_public_kmeans_baseline.py:37,427-441,484](scripts/run_public_kmeans_baseline.py#L37) | Dễ vô tình báo cáo init tốt hơn |
| M4 | `compute_external_metrics` raise nếu #cụm dự đoán ≠ K | [external_metrics.py:51-61](src/pipeline/external_metrics.py#L51-L61) | Run public có thể chết giữa chừng thay vì ghi NaN |
| M5 | Silhouette n×n tính 2 lần mỗi epoch cho unlabeled | [MVDEC_dense.py:738,778-784](src/representation_learning/MVDEC_dense.py#L738) | Chậm với 1400 epoch |
| M6 | Kết quả grid không có trong repo | `output/` chỉ có 3 file clusters | README:615-616 khẳng định sai sự thật |
| M7 | CSV grid thiếu `random_seed`, `fit_time_seconds`, `n_itemsets` | [experiments.py:95-123](src/pipeline/experiments.py#L95-L123) | Không audit được runtime/seed cấp dòng |
| M8 | `docs/*.pdf` bị comment trong `.gitignore` | `.gitignore` | 2 PDF bản quyền nằm trong repo định public + Zenodo DOI |
| M9 | Numeric Gower tính lại thừa mỗi candidate FFS | [clustering.py:114-118](src/pipeline/clustering.py#L114-L118) | Lãng phí lớn nhất của FFS |

---

### 🟢 LOW

- **L1** `H_FUSED_COLUMNS` 11 cột ([config.py:7](src/config.py#L7)) là dead config từ thời concat.
- **L2** `pyproject.toml` không khai báo tensorflow/torch, không pin version; README không hướng dẫn `uv sync --frozen`.
- **L3** `save_results` ghi lại toàn bộ CSV sau mỗi record ([io.py:59-91](src/pipeline/io.py#L59-L91)) → I/O O(n²).
- **L4** Draft lặp nguyên đoạn "The remainder of this paper is organized as follows"; numbering nhảy 3.2.1 → 3.3.2.
- **L5** `pickle.load` trên artifact ([data.py:311](src/pipeline/data.py#L311)) — chấp nhận được, nên nêu rõ trong README.

---

### ✅ Những điểm ĐÚNG, cần giữ

- **Ground truth không rò rỉ.** FP-Max, FFS, model selection chỉ dùng silhouette; `y` chỉ vào log và metric cuối.
- **ACC Hungarian đúng chuẩn** ([external_metrics.py:63-69](src/pipeline/external_metrics.py#L63-L69)).
- **Eigen direction đúng paper.** `sorted_eig` trả eigenvector cột, tăng dần; khớp DEKM Algorithm 1.
- **`count_aligned_assignment_changes`** viết lại sạch, có guard `previous < 0`.
- **DEKM reproduction tốt**: log REUTERS ACC 75.60 / NMI 59.35 vs paper 76.28 / 59.06.
- Fingerprint SHA-256 chuẩn hoá line-ending, ghi CSV atomic, manifest chống trộn dataset, artifact public tách theo seed, `delta_vs_mvdec` trên cùng biểu diễn — thiết kế tốt.
- **93/93 test pass** (66 non-TF + 27 TF).

---

## PHẦN 2 — PHÂN LOẠI

**A. Bug bắt buộc sửa:** C2, C3, C5, M1, M2, M4, M6, M8.

**B. Làm kết luận paper thiếu công bằng (phải sửa trước khi nộp):** C1, C4, H1,
H3, H4, H5, H6.

**C. Cải tiến tùy chọn:** H2/M9, M3, M5, M7, L1–L5.

---

## PHẦN 3 — FLOW ĐÁNH GIÁ ĐỀ XUẤT

### 3.1. Nguyên tắc xương sống

> **Một biểu diễn, một cây thước, mọi phương pháp.**

Mọi so sánh phải trả lời: *nếu giữ nguyên nhãn của baseline và chỉ đổi cách đo,
điểm có đổi không?* Nếu có, so sánh vô hiệu.

### 3.2. Public datasets (REUTERS-10K, 20NEWS, RCV1-10K)

- **Metric chính**: ACC + NMI (Hungarian), mean ± std trên **10 seed** (42–51), không phải 3.
- **Metric phụ**: ARI (chưa có trong đường chính — nên thêm), runtime.
- **Multi-seed full-pipeline**: mỗi seed train lại autoencoder từ đầu (public đã đúng, chỉ cần tăng số seed).
- **Ngân sách đồng nhất**: DEKM và MvDEC cùng `n_init=100`, cùng `update_interval`, cùng trần bước gradient, cùng tolerance 0.1%. Ghi vào manifest.
- **Statistical test**: Wilcoxon signed-rank theo cặp (method, dataset, seed); nếu ≥5 dataset thì Friedman + Nemenyi post-hoc. Báo p-value + effect size (Cliff's delta).
- **Ground truth**: chỉ ở bước cuối — đã đúng, giữ nguyên.

### 3.3. Private datasets (Tiki, Air Pollution)

Ba lớp báo cáo, không lẫn lộn:

**Lớp 1 — Native.** Mỗi phương pháp trong không gian của nó. Chỉ mô tả, không so
sánh chéo. Dùng `mvdec-evaluate-native`.

**Lớp 2 — Controlled mixed-space (SO SÁNH CHÍNH).** Cố định một biểu diễn hỗn
hợp (h_fused + FP-Max đã chọn), chấm tất cả nhãn (`MvDEC`, `MiMvDEC-noFFS`,
`MiMvDEC-FFS`, `K-Means`, `DEKM`) trên đúng ma trận đó. Báo cáo **Δ so với
MvDEC**, không phải điểm tuyệt đối (repo đã có `silhouette_delta_vs_mvdec`).
→ Table 6 phải viết lại theo cột Δ này.

**Lớp 3 — Common space (kiểm tra chéo).** Chấm mọi nhãn trên dữ liệu tiền xử lý
gốc, cả Euclid và Gower (`mvdec-evaluate-common`). Thước trung lập nhất.

**Quy tắc vàng**: nếu Lớp 2 và Lớp 3 mâu thuẫn → không tuyên bố ưu thế.

### 3.4. Protocol chọn feature (chống cherry-picking)

1. Tách chọn khỏi báo cáo: chọn config trên seed 0–4, báo cáo trên seed 5–14.
2. Ngưỡng FFS có ý nghĩa: thay `1e-6` bằng ngưỡng dựa trên std qua seed, hoặc permutation test.
3. Baseline cùng thước: baseline FFS = silhouette nhãn MvDEC **trên chính ma trận hỗn hợp của candidate đó** (sửa 1 dòng, đổi toàn bộ tính hợp lệ).
4. Báo cáo đường cong đầy đủ (mean ± std qua seed cho cả 171 config), không chỉ đỉnh.
5. Số feature được chọn phải đi kèm điểm số.

### 3.5. Ablation cô lập đúng đóng góp

| Bước | Biểu diễn | Thuật toán | Thước | Cô lập được |
|---|---|---|---|---|
| A0 | h_fused | K-Means | mixed Gower | điểm neo |
| A1 | h_fused | K-Prototypes | mixed Gower | ảnh hưởng thuật toán |
| A2 | h_fused + tất cả FP-Max | K-Prototypes | mixed Gower | ảnh hưởng FP-Max |
| A3 | h_fused + FP-Max đã FFS | K-Prototypes | mixed Gower | ảnh hưởng FFS |

Với air pollution, A2→A3 = 0 → nói thẳng trong paper FFS chỉ có tác dụng trên
Tiki, kèm giả thuyết vì sao.

Ablation phụ: symmetric vs asymmetric Gower (đã có), λ2=0 vs 1 (đã có CLI),
largest vs smallest eigen (đã có CLI).

---

## PHẦN 4 — THAY ĐỔI CODE TỐI THIỂU

| # | Thay đổi | File | Ước lượng |
|---|---|---|---|
| 1 | Baseline FFS = silhouette nhãn MvDEC trên ma trận hỗn hợp của candidate | `forward_selection.py`, `experiments.py`, `cli.py` | ~30 dòng — sửa C1, quan trọng nhất |
| 2 | Thêm seed vào tên artifact/CSV cho unlabeled datasets | `MVDEC_dense.py:1437,1243` | ~10 dòng — sửa C3 |
| 3 | Luôn đưa `lambda_kmeans/lambda_greedy` vào tên file (bỏ nhánh rỗng cho mặc định) | `MVDEC_dense.py:165-171` | ~5 dòng — sửa C5 |
| 4 | Regenerate artifact air-pollution bằng code hiện tại | chạy `MVDEC_dense.py AIRPOLLUTION` | 1 lệnh GPU — sửa C2 |
| 5 | Thống nhất `n_init=100` + `update_interval` + trần step giữa DEKM và MvDEC, ghi manifest | `DEKM_dense.py:103,164,95` | ~10 dòng — sửa H1 |
| 6 | Thống nhất `number_of_batches` dùng chung một hàm (ceil) | `MVDEC_dense.py:228`, `data.py:158` | ~5 dòng — sửa M1 |
| 7 | Thêm `random_seed`, `fit_time_seconds`, `n_itemsets` vào schema CSV grid | `experiments.py:95-123` | ~20 dòng — sửa M7 |
| 8 | Cache `compute_gower_distance(continuous_df)` một lần cho cả FFS | `clustering.py:114`, `forward_selection.py` | ~15 dòng — H2/M9 |
| 9 | Module mới `pipeline/statistics.py`: Wilcoxon + Friedman/Nemenyi + Cliff's delta | mới | ~120 dòng — sửa C4 |
| 10 | Prefix `--run-tag` cho pretrain weights + `log_history` | `MVDEC_dense.py`, `utils.py` | ~10 dòng — sửa M2 |
| 11 | Sửa README:615-616 và bổ sung `docs/*.pdf` vào `.gitignore` | `README.md`, `.gitignore` | 2 dòng — M6, M8 |

**Thứ tự đề nghị**: 1 → 3 → 2 → 4 → 5 (khoá tính hợp lệ trước) → 6, 10, 11 (bug
vặt) → 9 (thống kê) → 7, 8 (chất lượng).

---

## Ba việc quyết định số phận paper

1. Viết lại Table 6 theo `silhouette_delta_vs_mvdec` trên biểu diễn cố định.
2. Tự chạy lại toàn bộ baseline air-pollution thay vì chép từ Kassem 2025.
3. Tách seed chọn model khỏi seed báo cáo.
