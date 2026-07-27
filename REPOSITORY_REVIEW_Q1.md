# Review toàn diện repository theo chuẩn code review và phương pháp nghiên cứu Q1

## Quy ước phân loại

- **[BUG - BẮT BUỘC SỬA]**: lỗi thực thi, sai identity thuật toán, mất artifact, sai kết quả hoặc không thể tái lập.
- **[FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]**: code có thể chạy nhưng thiết kế thí nghiệm không cho phép quy kết causal/contribution như paper dự kiến.
- **[TÙY CHỌN]**: cải tiến engineering, reporting hoặc robustness không chặn kết luận chính nếu các lỗi bắt buộc đã được xử lý.

Review được thực hiện trên commit `5d431942178a9d139d1918f9b98451a913a6af26`, nhánh `dev`, với hai thay đổi có sẵn từ trước trong worktree: `src/representation_learning/MVDEC_dense.py` và `tests/test_mvdec_preprocessing.py`. Không file code nào được sửa trong quá trình review.

## Trạng thái xử lý C1 sau review - RESOLVED

C1 bên dưới mô tả đúng trạng thái tại commit được review. Worktree sau đó đã
chọn protocol chính `mvdec_dekm_consistent_v1`: kiến trúc/fusion theo MvDEC
2025, nhưng refinement theo DEKM 2021 với `L1=1`, `L2=0`, `L3=0`, `L4=1`,
eigenvalues tăng dần và chọn direction cuối/lớn nhất. Lý do bỏ `L2/L3` được ghi
đúng phạm vi: DEKM Fig. 4 cung cấp bằng chứng trên MNIST rằng tối ưu direction
cuối tốt hơn kéo toàn bộ transformed dimensions; đây không phải chứng minh tổng
quát rằng `L2 + L4` luôn kém hơn trên mọi dataset.

Primary target được đổi sang `frozen_snapshot` để khớp released implementation
của DEKM 2021; `selected_dimension_only` chỉ còn là custom ablation. Public giữ
tolerance `0.001`; Air Pollution/Tiki giữ `0.01` như một chủ ý thiết kế được lưu
trong protocol contract, không được claim là giá trị từ paper. Reduction hiện
tại của `L4` chưa thay đổi và được ghi rõ trong artifact để tách khỏi quyết định
fidelity tiếp theo. Vì vậy protocol này phải được gọi
**MvDEC-DEKM-consistent**, không phải literal MvDEC 2025 Eq. (11).

Producer và loader hiện dùng cùng contract: loader xác minh objective, loss
weights, eigen direction, target mode, tolerance theo dataset và SHA-256 của
protocol contract. Primary protocol từ chối tolerance khác; ablation phải dùng
`--protocol custom`. Artifact giữ `algorithm_family=MvDEC`, còn method label
trong artifact/assignment/summary là `MvDEC-DEKM-consistent`. Regression test
đã bao phủ chuỗi producer -> loader và trường hợp contract bị sửa. Run-id,
seed-specific output và custom-config collision được xử lý riêng trong trạng
thái C2 ngay sau đây.

## Trạng thái xử lý C2 sau review - RESOLVED

Mỗi MvDEC run hiện có `run_id = dataset + protocol_id + config_hash + seed` và
thư mục riêng theo dataset/protocol/config/seed. Artifact, assignments, pretrain
weights, final weights, training log và manifest không còn dùng chung path giữa
các seed. Full resolved config, source hash và protocol contract tạo SHA-256;
custom ablations khác config vì vậy đi vào config directory khác.

Run directory không rỗng bị từ chối mặc định; `--force` chỉ xóa các file thuộc
đúng run identity đã resolve, không xóa đệ quy thư mục. Manifest được ghi
`running`, chuyển thành `complete` sau khi lưu và hash toàn bộ outputs, hoặc
`failed` cùng loại/message lỗi nếu training thất bại. `mvdec_runs.csv` và
`mvdec_summary.csv` được dựng lại từ các manifest `complete`; failed runs vẫn
được giữ để audit nhưng không được đưa vào mean/std. Regression tests bao phủ
seed isolation, config isolation, overwrite guard, force và manifest status.

## Trạng thái xử lý C3 sau review - CODE RESOLVED, GPU RERUN PENDING

Các pickle/history/log/assignment MvDEC Air Pollution và Tiki tạo bởi protocol
cũ đã bị xóa; raw/preprocessed datasets và artifacts baseline không thuộc MvDEC
được giữ nguyên. CLI không còn default tới một pickle Tiki legacy: người chạy
phải truyền `--representation-path` trỏ tới `artifact.pkl` trong run directory.

Loader hiện từ chối mọi artifact mang fusion contract MvDEC nhưng thiếu
`protocol_id`, `run_id`, `config_hash` hoặc complete `manifest.json`. Khi load,
nó đối chiếu artifact với manifest, resolved config, protocol contract, seed,
đường dẫn và SHA-256 của artifact. Lệnh `mvdec-audit-runs` kiểm tra hashes của
toàn bộ outputs rồi reload từng complete artifact; failed manifests được báo
cáo nhưng không giả thành công.

Phần code/cleanup của C3 đã hoàn tất. C3 chỉ được đánh dấu hoàn toàn `RESOLVED`
sau khi Air Pollution và Tiki được train lại trên GPU theo các seed đã freeze,
chạy `mvdec-audit-runs` thành công và lưu các config directories mới. Repository
không còn artifact cũ để vô tình dùng làm bảng paper trong thời gian chờ rerun.

## REUTERS View2 architecture ablation - DIAGNOSTIC COMPLETE

Hai protocol chẩn đoán View2 trên REUTERS seed 42 đã được audit. Encoder
bottleneck cho fused `ACC=0.7643`, `NMI=0.6033`, nhưng View2 standalone giảm còn
`ACC=0.3843`, `NMI=0.0944`. Direct joint head gần Fig. 2 tiếp tục giảm View2 còn
`ACC=0.3701`, `NMI=0.0943`, với fused `ACC=0.7621`, `NMI=0.6002`. Cả hai là
negative ablations, không được chọn làm kiến trúc chính và không mở rộng thêm
seed. Fig. 2 của MvDEC 2025 vẫn mâu thuẫn với Eq. (3)-(5) về terminal head và
với Eq. (4) về fusion. Toàn bộ provenance, checksum và bảng đối chiếu paper được
ghi tại `MVDEC_REUTERS_ARCHITECTURE_DIAGNOSIS.md`.

Fusion-sensitivity protocol
`mvdec_2025_view2_direct_23_split_concat_v1` đã được implement để giữ nguyên
direct View2 head nhưng thay Eq. (4) average 10D bằng latent concatenation 20D.
Protocol, config hash, artifact fusion contract và output directory độc lập;
loader kiểm tra trực tiếp `h_fused = concatenate(h_view1, h_view2)`. Đây chỉ là
diagnostic arm seed 42, không phải phương pháp được chọn bằng ground truth.

## Findings - Critical

### C1. Mặc định đang được gọi là MvDEC không tối ưu objective của MvDEC 2025

**Loại:** [BUG - BẮT BUỘC SỬA] và [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- Paper MvDEC 2025, Eq. (11), p.8 ghi rõ `L = L1 + L2 + L3 + L4`, với `L1` reconstruction hai view, `L2` K-means, `L3 = Tr(V S_w V^T)` và `L4` greedy centroid target.
- Code hiện đặt `DEFAULT_LAMBDA_KMEANS = 0.0` và `lambda_orthonormal = 0.0`, nên default thực tế là `L1 + L4`, không phải Eq. (11): `src/representation_learning/MVDEC_dense.py:39`, `src/representation_learning/MVDEC_dense.py:43`, `src/representation_learning/MVDEC_dense.py:47`, `src/representation_learning/MVDEC_dense.py:1077`.
- CLI chỉ expose `--lambda-kmeans` và `--lambda-greedy`, không có cách bật `L3`: `src/representation_learning/MVDEC_dense.py:1337`, `src/representation_learning/MVDEC_dense.py:1346`.
- Test xác nhận `L3` trace-equivalent bằng đúng squared distance của `L2`: `tests/test_mvdec_losses.py:53`. Vì vậy Eq. (11) với hệ số 1 có effective centroid gradient bằng `2 x L2`; `--lambda-kmeans 1` vẫn không phải literal Eq. (11).
- README lại nói joint training kết hợp reconstruction, K-means và greedy: `README.md:251`, trong khi command mặc định tại `README.md:153` không truyền `--lambda-kmeans 1`.
- MvDEC 2025 p.9 còn gọi `Proposed method_View1 = DEKM_F` (full-batch DEKM), trong khi code cập nhật từng balanced mini-batch: `src/representation_learning/MVDEC_dense.py:1020`, `src/representation_learning/MVDEC_dense.py:1085`.

**Hậu quả phương pháp luận:** kết quả default không được phép ghi là faithful MvDEC 2025. Nó là một hybrid/ablation mới. So sánh DEKM - MvDEC - MiMvDEC hiện không có identity thuật toán ổn định.

**Bắt buộc:** tạo protocol/objective enum bất biến, ví dụ `mvdec_2025_literal`, `mvdec_dekm_consistent`, `mvdec_l1_l4_ablation`; lưu toàn bộ loss weights; tên method, filename và manifest phải phản ánh đúng protocol. Nếu bỏ `L3` vì trùng `L2`, phải dùng effective weight tương đương và giải thích toán học, không gọi `lambda_kmeans=1` là literal Eq. (11).

### C2. Multi-run và ablation bị ghi đè, chỉ artifact của run cuối còn tồn tại

**Loại:** [BUG - BẮT BUỘC SỬA]

- Với Air Pollution/Tiki, vòng `--runs N` dùng cùng `run_artifact_path` cho mọi seed: `src/representation_learning/MVDEC_dense.py:1423`, `src/representation_learning/MVDEC_dense.py:1437`, `src/representation_learning/MVDEC_dense.py:1483`.
- Assignments của mọi run cũng dùng cùng path theo dataset/protocol, không có seed: `src/representation_learning/MVDEC_dense.py:1232`, `src/representation_learning/MVDEC_dense.py:1243`.
- Final weights có eigen/target/loss tag nhưng không có seed: `src/representation_learning/MVDEC_dense.py:1162`. Pretrain weights cũng ghi đè theo dataset: `src/representation_learning/MVDEC_dense.py:703`, `src/representation_learning/MVDEC_dense.py:722`.
- Với public datasets, pickle path chỉ chứa method/dataset/seed, không chứa eigen direction, target mode, loss weights, tolerance hoặc max epochs: `src/representation_learning/MVDEC_dense.py:1462`. Chạy ablation khác với cùng seed sẽ ghi đè artifact và assignments cũ.
- `mvdec_runs.csv` và `mvdec_summary.csv` dùng tên cố định: `src/representation_learning/MVDEC_dense.py:1553`, `src/representation_learning/MVDEC_dense.py:1578`.
- README tuyên bố các mode không overwrite nhau: `README.md:177`, `README.md:182`; tuyên bố này không đúng cho multi-run private và public ablations.

**Hậu quả phương pháp luận:** mean của nhiều runs nằm trong log nhưng chỉ run cuối có representation/assignment. Không thể audit, chạy downstream full-pipeline theo seed, hoặc xác nhận mean +/- std. Public ablation có thể âm thầm thay nội dung của cùng filename.

**Bắt buộc:** mỗi run phải có immutable `run_id = dataset + method_protocol + config_hash + seed`; tạo thư mục riêng cho seed; mặc định từ chối overwrite; summary chỉ được sinh bằng cách join các run manifests hoàn chỉnh.

### C3. Artifact đang track không tương thích với code hiện tại; Tiki ablation còn trộn seed, batching và loss

**Loại:** [BUG - BẮT BUỘC SỬA] và [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- `data/preprocessed_data/airpollution_demvk_fused_representation.pkl` đang track dùng legacy `fusion_contract=mvdec2025_figure_output_average`, `h_fused` 23 chiều và thiếu metadata preprocessing/eigen/stop mới. Loader chủ động từ chối contract này: `src/pipeline/data.py:321`, `src/pipeline/data.py:330`.
- Command được ghi trong runbook dùng chính artifact đó: `RUNBOOK_GPU_TO_LOCAL.md:68`, `RUNBOOK_GPU_TO_LOCAL.md:75`, `RUNBOOK_GPU_TO_LOCAL.md:90`. Kiểm tra read-only thực tế trả về: `ValueError: This artifact uses the legacy full-view-output average`.
- Default Tiki pickle đang track có `seed=44`, `batches_per_epoch=8`, không có `refinement_batching_policy`, và `loss_weights.kmeans=1.0`. Code hiện tính Tiki thành 7 balanced batches và default `lambda_kmeans=0`: `src/representation_learning/MVDEC_dense.py:228`, `src/representation_learning/MVDEC_dense.py:257`, `src/representation_learning/MVDEC_dense.py:43`.
- Frozen-snapshot pickle đang track có `seed=42`, 7 batches và `lambda_kmeans=0`. Log xác nhận selected-dimension dùng seeds 42/43/44 nhưng chỉ seed 44 còn trong pickle: `log_history/TIKI_LARGEST_EIGEN_SELECTED_DIMENSION_ONLY.csv:34`, `log_history/TIKI_LARGEST_EIGEN_SELECTED_DIMENSION_ONLY.csv:74`, `log_history/TIKI_LARGEST_EIGEN_SELECTED_DIMENSION_ONLY.csv:114`. Frozen snapshot chỉ có seed 42: `log_history/TIKI_LARGEST_EIGEN_FROZEN_SNAPSHOT.csv:49`.
- Validation chặt về batching/stop/preprocessing chỉ áp dụng cho `AIRPOLLUTION`: `src/pipeline/data.py:110`; Tiki artifact stale vẫn pass.

**Hậu quả phương pháp luận:** artifact Air Pollution không chạy; Tiki target-mode comparison đồng thời đổi target mode, seed, L2 weight và batching implementation. Không thể quy chênh lệch cho target mode hoặc eigen semantics.

**Bắt buộc:** regenerate toàn bộ tracked artifacts bằng một commit/protocol đã freeze; xóa hoặc chuyển legacy artifacts sang thư mục archival có schema riêng; áp cùng validation schema cho mọi dataset; không dùng các artifact Tiki hiện tại cho ablation paper.

### C4. FP-Max/FFS và grid search chọn rồi báo cáo trên cùng dữ liệu, tạo selection bias trực tiếp

**Loại:** [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- Grid mặc định gồm 3 strategies x 3 bin counts x 19 supports = 171 configurations: `src/pipeline/fpmax.py:18`, `src/pipeline/fpmax.py:19`, `src/pipeline/experiments.py:2153`, `src/pipeline/experiments.py:2334`.
- `mvdec-evaluate-best` mặc định lấy `idxmax()` của score trên chính grid đó: `src/pipeline/best_evaluation.py:68`, `src/pipeline/best_evaluation.py:90`, `src/pipeline/best_evaluation.py:98`.
- FFS tiếp tục thử nhiều candidate trên cùng toàn bộ dataset và dừng với improvement chỉ `1e-6`: `src/pipeline/forward_selection.py:170`, `src/pipeline/forward_selection.py:177`, `src/pipeline/forward_selection.py:269`.
- Mỗi candidate đồng thời thay feature space, distance matrix và assignments: `src/pipeline/forward_selection.py:100`, `src/pipeline/forward_selection.py:105`.
- Với private datasets không có ground truth, Silhouette vừa là objective chọn strategy/bin/support/feature/backend parameters vừa là metric kết luận cuối.

**Hậu quả phương pháp luận:** reported Silhouette là in-sample maximum sau hàng trăm/hàng nghìn phép thử, không phải estimate độc lập. FFS gần như được định nghĩa để tăng chính metric đang dùng để chứng minh đóng góp; đây là circular validation và dễ bị xem là cherry-picking trong review Q1.

**Bắt buộc:** chọn config/feature trên development split hoặc inner resampling; freeze trước outer evaluation; dùng out-of-sample Silhouette và clustering stability; dừng FFS bằng predeclared margin/CI hoặc one-standard-error rule, không bằng epsilon số học.

### C5. `mvdec-evaluate-best --seeds` không phải full-pipeline multi-seed

**Loại:** [BUG VỀ CLAIM] và [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- Evaluator load đúng một MvDEC artifact, rebuild một feature set cố định, rồi chỉ thay seed của K-Prototypes/Intuitive: `src/pipeline/best_evaluation.py:132`, `src/pipeline/best_evaluation.py:156`, `src/pipeline/best_evaluation.py:688`, `src/pipeline/best_evaluation.py:706`.
- `h_fused`, MvDEC assignments, mined feature definition và MvDEC ACC/NMI reference đều bất biến qua các seed downstream: `src/pipeline/best_evaluation.py:250`, `src/pipeline/best_evaluation.py:271`, `src/pipeline/best_evaluation.py:290`.
- README thừa nhận variance chỉ đo final-model stability: `README.md:582`; do đó không thể dùng output này như full-pipeline mean +/- std.

**Hậu quả phương pháp luận:** std bỏ qua phần variance lớn nhất: neural initialization, pretraining shuffle, MvDEC refinement, FP-Max mining và FFS. Delta so với MvDEC không paired theo seed vì baseline là một hằng số.

**Bắt buộc:** một seed phải chạy xuyên suốt raw/preprocessed input -> DEKM/MvDEC -> FP-Max -> FFS -> clustering -> final evaluation. Mỗi method dùng cùng seed list và mỗi delta phải ghép đúng `dataset_hash + seed + protocol`.

### C6. Ablation hiện tại không cô lập đóng góp FP-Max hoặc FFS

**Loại:** [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- MvDEC dùng K-means trên `h_fused`; MiMvDEC without FFS đổi đồng thời sang K-Prototypes, mixed features và asymmetric Gower scoring: `src/representation_learning/MVDEC_dense.py:742`, `src/pipeline/experiments.py:4247`, `src/pipeline/clustering.py:248`.
- Controlled evaluator chấm MvDEC labels trong chính mixed feature space do MiMvDEC-selected config tạo ra: `src/pipeline/best_evaluation.py:250`, `src/pipeline/best_evaluation.py:292`.
- With FFS và without FFS có thể tự chọn hai best rows khác nhau bằng `idxmax()`, nên còn thay strategy, bins và support: `src/pipeline/best_evaluation.py:90`.
- Không có arm `h_fused + cùng downstream backend nhưng không FP-Max`, cũng không có arm `discretized one-hot bins nhưng không maximal itemsets`.

**Hậu quả phương pháp luận:** gain MvDEC -> MiMvDEC bundles backend, distance, feature expansion, hyperparameter search và FP-Max. Gain without FFS -> with FFS bundles FFS với khả năng chọn config khác. Không thể viết “FP-Max đóng góp X” hoặc “FFS đóng góp Y”.

**Bắt buộc:** dùng ablation ladder paired trong cùng seed và cùng config: MvDEC K-means -> same-backend/no-FP -> one-hot bins only -> FP-Max all -> FP-Max + FFS. So sánh FFS phải dùng cùng candidate itemsets, strategy, bins, support, backend và restart budget.

### C7. Exact Gower dense `O(N^2)` làm public full pipeline không khả thi về RAM

**Loại:** [BUG KHẢ NĂNG CHẠY - BẮT BUỘC SỬA CHO PUBLIC]

- Numeric Gower materialize ma trận dense: `src/pipeline/clustering.py:86`.
- Asymmetric binary tạo thêm intersection, union, mismatch và distance matrices `N x N`: `src/pipeline/clustering.py:143`.
- Best evaluator giữ đồng thời asymmetric và symmetric matrices: `src/pipeline/best_evaluation.py:250`.
- Local DEKM release được quan sát có 20NEWS `18,057 x 2,000`; một ma trận float64 đơn đã khoảng 2.61 GB (2.43 GiB). Theo 18,846 rows của DEKM paper, một matrix khoảng 2.84 GB. Peak với intermediates dễ vượt 10-15 GB cho một candidate; candidate/outer multiprocessing nhân mức RAM này lên nhiều lần.
- Air Pollution 3,765 rows cần khoảng 113 MB cho một matrix float64; một FFS candidate có thể dùng vài trăm MB. Runbook còn đề xuất outer 20 x inner 10 workers: `RUNBOOK_GPU_TO_LOCAL.md:101`, `RUNBOOK_GPU_TO_LOCAL.md:104`.

**Hậu quả phương pháp luận:** MiMvDEC full public benchmark, đặc biệt 20NEWS, có nguy cơ OOM hoặc không thể hoàn tất; việc chỉ báo public MvDEC/DEKM rồi MiMvDEC trên subset sẽ phá fairness nếu không khai báo.

**Bắt buộc:** public ACC/NMI không cần full Gower; dùng ACC/NMI làm primary và internal metric chỉ trên fixed label-blind subsample, hoặc triển khai chunked/approximate silhouette, memmap/k-nearest silhouette. Thêm memory preflight và cấm nested process pools vượt budget.

## Findings - High

### H1. DEKM thay `n_init` bằng `2 * n_iter_` và có thể lưu final state stale

**Loại:** [BUG - BẮT BUỘC SỬA]

- DEKM bắt đầu `n_init=100` nhưng sau mỗi K-means gán `n_init = 2 * ans_kmeans.n_iter_`: `src/representation_learning/DEKM_dense.py:92`, `src/representation_learning/DEKM_dense.py:103`. Số random restarts và số convergence iterations không cùng ý nghĩa.
- Nếu chạm cuối `range(140 * 100)`, model còn được gradient-update sau checkpoint cuối nhưng `H`/assignment không được recompute; function vẫn return stale state: `src/representation_learning/DEKM_dense.py:95`, `src/representation_learning/DEKM_dense.py:143`, `src/representation_learning/DEKM_dense.py:157`.
- Final weights chỉ được save khi convergence branch chạy; không có `stop_reason`: `src/representation_learning/DEKM_dense.py:139`.

**Hậu quả:** DEKM có K-means search budget không ổn định và artifact có thể không tương ứng final encoder. So sánh với MvDEC fixed `n_init=100` không công bằng.

### H2. DEKM và MvDEC không có cùng refinement/compute budget

**Loại:** [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- DEKM refresh K-means mỗi 10 mini-batches và luôn có 14,000 gradient steps tối đa: `src/representation_learning/DEKM_dense.py:95`, `src/representation_learning/DEKM_dense.py:164`.
- MvDEC refresh sau một full epoch và đặt max steps bằng `1400 x batches_per_epoch`: `src/representation_learning/MVDEC_dense.py:842`, `src/representation_learning/MVDEC_dense.py:844`.
- Với Reuters khoảng 39 batches/epoch, MvDEC có khoảng 54,600 steps; với local 20NEWS khoảng 71 batches/epoch, khoảng 99,400 steps, lớn hơn DEKM nhiều lần.

**Hậu quả:** improvement có thể đến từ optimization budget, refresh policy và số K-means restarts, không chỉ multi-view/loss.

**Bắt buộc:** báo hai protocol riêng: `DEKM-release-faithful` và `controlled-budget`. Controlled comparison phải cố định số gradient examples/steps, refresh count, K-means `n_init`, tolerance và max wall-clock hoặc compute budget.

### H3. Ground truth bị peek trong training loop public

**Loại:** [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- DEKM tính/log ACC và NMI tại mọi refresh: `src/representation_learning/DEKM_dense.py:131`, `src/representation_learning/DEKM_dense.py:137`.
- MvDEC gọi external metrics cho fused embedding và greedy target mỗi epoch: `src/representation_learning/MVDEC_dense.py:966`, `src/representation_learning/MVDEC_dense.py:998`.
- Metrics không đi trực tiếp vào gradient hoặc auto-stop, nhưng chúng hiển thị liên tục cho researcher.

**Hậu quả:** vi phạm protocol “ground truth chỉ dùng ở đánh giá cuối” và tạo researcher degrees of freedom để dừng, chọn seed/config/epoch hậu nghiệm.

**Bắt buộc:** train API không nhận `y`; chỉ export row-ordered assignments. Một evaluator tách biệt, chạy sau khi artifact/config được freeze, mới được mở/join labels.

### H4. Legacy processed public path dùng labels để sampling và chọn feature

**Loại:** [BUG LEAKAGE NẾU DÙNG] và [FAIRNESS]

- 20NEWS được lấy đúng 100 documents mỗi class và chọn top-2,000 features bằng chi-square với `y`: `scripts/fetch_public_datasets.py:699`, `scripts/fetch_public_datasets.py:703`.
- RCV1 được label-balance 2,500/class và cũng chi-square với `y`: `scripts/fetch_public_datasets.py:732`, `scripts/fetch_public_datasets.py:785`.
- K-means CLI vẫn expose `--source processed`: `scripts/run_public_kmeans_baseline.py:85`, `scripts/run_public_kmeans_baseline.py:469`.

**Hậu quả:** bất kỳ kết quả main-table nào dùng processed path đều không phải unsupervised label-blind benchmark. Default `dekm-release` an toàn hơn nhưng mode leakage vẫn tồn tại và dễ bị trộn vào output.

**Bắt buộc:** cấm processed supervised path trong main comparison, hoặc gắn rõ `label-informed preprocessing`; nếu cần feature selection, dùng rule unsupervised hoặc fit trong nested protocol mà labels chỉ dùng final.

### H5. Seed defaults và GPU determinism không đủ tái lập

**Loại:** [BUG TÁI LẬP - BẮT BUỘC SỬA]

- DEKM và MvDEC mặc định `--seed=None`: `src/representation_learning/DEKM_dense.py:169`, `src/representation_learning/MVDEC_dense.py:1295`.
- Khi seed `None`, `set_random_seed` không làm gì: `src/representation_learning/MVDEC_dense.py:188`.
- Unlabeled loader tự sinh shuffle seed từ NumPy nhưng không return/lưu giá trị thực: `src/representation_learning/MVDEC_dense.py:377`, `src/representation_learning/MVDEC_dense.py:380`.
- Chỉ `np.random.seed` và `tf.random.set_seed` được gọi; không seed Python `random`, không dùng `keras.utils.set_random_seed`, không bật deterministic TensorFlow ops/CUDA.
- Command Air Pollution đầu tiên trong README/runbook chạy 3 runs nhưng không truyền seed: `README.md:153`, `RUNBOOK_GPU_TO_LOCAL.md:19`.

**Hậu quả:** artifact có thể ghi seed `None` nhưng không có cách tái tạo. GPU results vẫn có thể khác dù seed explicit.

### H6. Public dataset contract không xác thực sample count hoặc raw-file identity

**Loại:** [BUG TÁI LẬP - BẮT BUỘC SỬA] và [FAIRNESS]

- `PublicDatasetSpec` chỉ giữ K và 2,000 features, không có expected rows, label distribution hoặc raw hashes: `src/pipeline/public_data.py:11`, `src/pipeline/public_data.py:31`.
- Validation chỉ kiểm tra width, X/y length, finite và số unique labels: `src/pipeline/public_data.py:174`, `src/pipeline/public_data.py:184`.
- Unit tests chấp nhận Reuters chỉ 4 rows và 20NEWS chỉ 20 rows: `tests/test_public_data.py:9`, `tests/test_public_data.py:30`.
- Quan sát local: Reuters là `10,000 x 2,000`, nhưng 20NEWS là `18,057 x 2,000`; DEKM 2021 Table II ghi 18,846. MvDEC 2025 Table 1 lại ghi 20NEWS 2,000 và Reuters “210”, trong khi Table 2 copy baseline full-dataset từ DEKM.
- RCV1 phụ thuộc scikit-learn cache ngoài repo và chọn top features on-the-fly, không lưu cache hash/feature indices: `src/pipeline/public_data.py:113`, `src/pipeline/public_data.py:126`, `src/pipeline/public_data.py:131`.

**Hậu quả:** cùng tên dataset có thể là ba population khác nhau. Current results có thể fair nội bộ nếu mọi method dùng cùng local arrays, nhưng không được gọi direct replication của sample contract trong hai paper.

**Bắt buộc:** version dataset contract bằng expected shape, raw/source file hashes, processed array hash, ordered sample IDs, label schema và feature indices. Với 20NEWS, ghi rõ “DEKM release artifact 18,057 rows” nếu đó là contract chọn; không silently gọi nó 18,846. Materialize RCV1 processed X/y một lần và hash.

### H7. “Multi-view” đang là hai architecture cùng nhìn một X; view semantics chưa được freeze

**Loại:** [FAIRNESS / PAPER CORRESPONDENCE]

- Hai model đều nhận `Input(shape=(input_shape,))`: `src/representation_learning/MVDEC_dense.py:539`, `src/representation_learning/MVDEC_dense.py:571`.
- Cả hai được pretrain từ cùng dataset `(x, x)`: `src/representation_learning/MVDEC_dense.py:1466`.
- MvDEC 2025 tự mâu thuẫn: prose nói hai “specific sets of attributes”, Eq. (3) và Fig. 2 lại cho cả hai cùng input 13 features. Prose nói average encoded representations, Fig. 2 hiển thị output 23 = latent + reconstruction.
- Code chọn interpretation `architecture views on same X` và fuse latent heads. Đây là lựa chọn hợp lý nhưng không phải fact duy nhất suy ra từ paper.

**Hậu quả:** không thể viết như thể code đã tái lập natural feature views. Với Tiki, architecture widths/K/latent còn là thiết kế mới hoàn toàn, không thuộc hai paper.

**Bắt buộc:** manifest lưu `view_definition`, exact feature lists và fusion contract. Manuscript phải gọi rõ “architecture-view interpretation”; nếu thử feature-view interpretation, đó là sensitivity protocol riêng.

### H8. Euclidean, Gower và ba evaluation spaces đang dễ bị diễn giải như cùng một thang đo

**Loại:** [FAIRNESS - BẮT BUỘC TRƯỚC KHI KẾT LUẬN]

- Artifact score là Euclidean Silhouette trên `h_fused`, loader còn tính numeric-Gower score khác: `src/pipeline/data.py:393`, `src/pipeline/data.py:394`.
- Native evaluator tính numeric Gower trong representation riêng của từng method: `src/pipeline/native_evaluation.py:283`, `src/pipeline/native_evaluation.py:294`.
- Common evaluator tính Euclidean và Gower trên cùng original numeric data: `src/pipeline/common_evaluation.py:235`.
- Best evaluator tính asymmetric/symmetric mixed Gower trên feature set selected của MiMvDEC: `src/pipeline/best_evaluation.py:250`.
- README gọi native mixed comparison là primary và common-space là optional: `README.md:585`, `README.md:587`.

**Kết luận phương pháp luận:**

1. Euclidean Silhouette và Gower Silhouette không được so trực tiếp về magnitude.
2. Cùng tên numeric Gower nhưng trên representation/dimension khác nhau vẫn không phải controlled between-method comparison; native scores chỉ là within-method diagnostics.
3. Controlled mixed-space là hợp lý để so assignments nếu và chỉ nếu mọi assignment được chấm trên đúng cùng một frozen matrix/distance. Current evaluator làm được điều này cho MvDEC-vs-một MiMvDEC config, nhưng không cho hai best configs with/without FFS khác nhau.
4. Common-space là direct assignment comparison công bằng nhất, nhưng chỉ đo geometry của common original space; nó không thay thế external validation.
5. Common Euclidean hiện dùng dữ liệu nguyên trạng: `src/pipeline/common_evaluation.py:237`. Air Pollution có scale/units rất khác nhau; Tiki winsorize sau standardization nên variance không còn đúng 1. Euclidean phải dùng một frozen standardized common matrix, còn Gower nên là primary private metric.

### H9. K-Prototypes không tối ưu asymmetric Gower đang được dùng để chọn và kết luận

**Loại:** [FAIRNESS / ALGORITHM-DISTANCE MISMATCH]

- K-Prototypes fit numeric Euclidean + categorical matching trên binary columns cast object: `src/pipeline/clustering.py:228`, `src/pipeline/clustering.py:267`, `src/pipeline/clustering.py:274`.
- Labels sau đó được chọn theo custom asymmetric Gower Silhouette: `src/pipeline/clustering.py:262`, `src/pipeline/clustering.py:276`, `src/pipeline/clustering.py:291`.
- Symmetric/asymmetric best-evaluation ablation dùng cùng labels; nó chỉ là metric sensitivity, không phải clustering-distance ablation.

**Hậu quả:** backend có thể tạo clusters nhờ shared `0-0` matches trong categorical objective nhưng evaluator lại bỏ shared absences. Không được mô tả K-Prototypes như đang tối ưu proposed asymmetric mixed distance.

**Bắt buộc trước claim distance contribution:** hoặc dùng clustering backend thực sự tối ưu cùng distance (ví dụ k-medoids/CLARA với contract đó), hoặc gọi rõ “fit objective” và “evaluation distance” là hai contract khác; chạy sensitivity với backend phù hợp.

### H10. Singleton cluster bị loại sau khi đã chọn best init/candidate

**Loại:** [BUG - BẮT BUỘC SỬA]

- `run_kprototypes` chọn init có Silhouette cao nhất mà không loại singleton trước: `src/pipeline/clustering.py:265`, `src/pipeline/clustering.py:291`.
- FFS cũng chọn candidate/init cao nhất trước: `src/pipeline/forward_selection.py:121`, `src/pipeline/forward_selection.py:150`, `src/pipeline/forward_selection.py:254`.
- Chỉ wrapper sau cùng mới đánh failed singleton: `src/pipeline/experiments.py:3642`, `src/pipeline/experiments.py:4506`.

**Hậu quả:** một init/candidate singleton có score cao có thể che một nghiệm hợp lệ thấp hơn; cả grid row hoặc đường FFS hợp lệ bị mất. Native Intuitive lại lọc singleton trước selection, nên backend treatment không nhất quán.

**Bắt buộc:** validation cluster count/min size phải chạy trước khi candidate được đưa vào argmax; thêm regression test có một singleton-high-score và một valid-lower-score.

### H11. CSV/pickle/manifest chưa đủ provenance để tái lập hoặc chống resume mixing

**Loại:** [BUG TÁI LẬP - BẮT BUỘC SỬA]

- Shared experiment manifest chỉ lưu dataset, data hash, representation hash, K, K-Prototypes n_init, một random state và distance contract: `src/pipeline/experiment_context.py:69`, `src/pipeline/experiment_context.py:78`.
- Nó không lưu git commit/dirty hash, full grid, backend, init strategy, FFS threshold, code/schema version theo method, dependency lock hash, hardware hay stage runtime.
- Base result schemas không có per-row seed, runtime, labels hash, itemset support, bin edges/effective bins hoặc FFS trace: `src/pipeline/experiments.py:95`, `src/pipeline/experiments.py:110`.
- `fit_time_seconds` và `labels_hash` đã có trong object clustering nhưng bị bỏ khi ghi CSV: `src/pipeline/clustering.py:33`, `src/pipeline/clustering.py:298`, `src/pipeline/experiments.py:3633`.
- Public artifact schema của DEKM/K-means chỉ lưu seed, K, labels, true labels, ACC/NMI và source hash; thiếu architecture, optimizer, stop, n_init history, runtime và environment: `src/pipeline/public_artifacts.py:83`.
- Best-evaluation output không lưu results CSV hash, score column/selected score, representation hash hay stage time: `src/pipeline/best_evaluation.py:720`.
- Tiki producer có nested source hash nhưng đặt top-level `source_sha256=None`; loader/best evaluator đọc top-level nên output hash rỗng: `src/representation_learning/MVDEC_dense.py:1436`, `src/pipeline/data.py:433`, `src/pipeline/best_evaluation.py:722`.

**Hậu quả:** resume có thể trộn rows trước/sau thay đổi code/default/library dù artifact hash không đổi. CSV không đủ để độc lập rebuild exact feature selection và audit runtime.

### H12. Clean checkout không tái lập được public/Air pipeline và README overstates availability

**Loại:** [BUG DATA AVAILABILITY / REPRODUCIBILITY]

- Exact DEKM datasets bị ignore và không có downloader pinned cho Google Drive release: `.gitignore:39`, `.gitignore:40`.
- Không có tracked `output/public_benchmark`, MiMvDEC grid results hoặc experiment manifests. README hiện nói rõ paper-ready outputs phải được regenerate và audit, không còn tuyên bố các output đi kèm tái tạo manuscript.
- Các assignment/reclustering CSV legacy trong `output/` đã bị xóa theo C3; repository vẫn chưa kèm các kết quả paper-ready được liệt kê trong workflow README.
- RCV1 phụ thuộc cache ngoài repo: `src/pipeline/public_data.py:113`.
- Air repository chỉ bắt đầu từ `data_demvk.csv`; parser public chỉ copy local preprocessed matrix, không tái tạo EEA/Sentinel/OSM/traffic spatial preprocessing: `scripts/fetch_public_datasets.py:799`, `scripts/fetch_public_datasets.py:809`.

**Hậu quả:** reviewer không thể chạy clean-room reproduction của public results hoặc raw-to-feature Air Pollution pipeline. Chỉ Tiki có raw workbook và regression preprocessing tương đối đầy đủ.

### H13. TensorFlow/GPU environment không được pin và khác xa DEKM release

**Loại:** [BUG TÁI LẬP - BẮT BUỘC SỬA]

- `pyproject.toml` không chứa TensorFlow/PyTorch và direct dependencies chủ yếu không có version range: `pyproject.toml:6`, `pyproject.toml:7`.
- README/runbook cài latest TensorFlow bằng `uv pip install tensorflow` hoặc `uv add tensorflow`: `README.md:136`, `README.md:147`, `RUNBOOK_GPU_TO_LOCAL.md:17`.
- `uv.lock` hiện pin local pipeline nhưng không pin TensorFlow.
- DEKM release ghi TensorFlow 2.4.1, scikit-learn 0.23.2, NumPy 1.19.5, SciPy 1.2.1: `external_repos/DEKM/README.md:57`.
- Môi trường review thực tế là TensorFlow 2.21.0, scikit-learn 1.9.0, NumPy 2.4.6, SciPy 1.17.1.

**Hậu quả:** K-means behavior, Keras initializers/optimizers, serialization và GPU kernels có thể khác. Không thể gọi exact DEKM release reproduction.

**Bắt buộc:** tạo environment/lock riêng cho `dekm_release_faithful` và `modern_controlled`; pin TensorFlow/CUDA/cuDNN, base image, Python và dependency hashes; artifact lưu versions/hardware/determinism flags.

### H14. K và preprocessing choices chưa có protocol selection công bằng

**Loại:** [FAIRNESS]

- Tiki hardcode `K=5` nhưng repository chỉ ghi “as requested”, không có data-driven/domain rationale hoặc K sensitivity: `src/representation_learning/MVDEC_dense.py:80`, `scripts/fetch_public_datasets.py:863`.
- Air hardcode `K=4`: `src/representation_learning/MVDEC_dense.py:70`. Paper chọn bằng visual elbow trên proposed method, K=2..9, nhưng không có rule định lượng/nested validation.
- Public K bằng số ground-truth classes: `src/pipeline/public_data.py:31`. Đây là paper-standard oracle-K protocol, nhưng không phù hợp tuyên bố “ground truth chỉ dùng cuối” nếu không ghi ngoại lệ.
- Tiki standardize -> winsorize -> DBSCAN remove 22/1,822 rows trước clustering: `src/data_preprocessing/pipeline.py:57`, `src/data_preprocessing/pipeline.py:60`, `tests/test_data_preprocessing.py:50`.

**Hậu quả:** K và data filtering có thể tối ưu geometry trước khi so methods. DBSCAN removal làm Silhouette lạc quan hơn dù áp chung methods.

**Bắt buộc:** công bố hai protocol public: `oracle-K paper replication` và `label-blind estimated-K`. Với private, chọn một K chung trên discovery data bằng predeclared stability/elbow rule rồi freeze cho mọi method; báo sensitivity K. Tiki báo ablation giữ toàn bộ rows vs DBSCAN-filtered.

### H15. Native/common evaluators chưa bắt buộc cùng dataset identity, sample IDs và seed set

**Loại:** [BUG ALIGNMENT / FAIRNESS]

- Native pickle loader không đối chiếu dataset/source-data hash giữa methods; aggregate chỉ group theo method string: `src/pipeline/native_evaluation.py:130`, `src/pipeline/native_evaluation.py:323`.
- Common row-level loader chỉ nhận `orig_index` hoặc `original_index`, không dùng `sample_index` mà public assignments thực sự lưu: `src/pipeline/common_evaluation.py:98`, `src/pipeline/public_artifacts.py:55`.
- Nếu CSV public bị reorder sau export, common evaluator sẽ silently dùng row order hiện tại.
- Serialized assignments chỉ kiểm tra số rows/seed uniqueness, không đối chiếu upstream dataset hash của từng source: `src/pipeline/common_evaluation.py:120`.

**Hậu quả:** có thể aggregate methods từ dataset version/seed set khác nhau hoặc misalign rows mà không fail fast.

**Bắt buộc:** mọi assignment source phải mang `dataset_id`, `data_sha256`, ordered `sample_id` hash, method protocol, seed và assignment hash; evaluator join/validate bằng IDs, không dựa vào row count.

### H16. Air/Tiki interpretation hiện là descriptive, không phải external validation

**Loại:** [FAIRNESS / RESEARCH CLAIM]

- MvDEC 2025 Table 4 dùng `NO2 Value` và neighbor-NO2 features làm input; p.16-17 sau đó dùng chính NO2 để “validate” high/low pollution clusters.
- Repository profile/interpretation cũng tổng hợp lại raw/preprocessed features đã tham gia clustering: `src/pipeline/best_evaluation.py:587`, `src/pipeline/best_evaluation.py:603`.
- Air features có spatial autocorrelation mạnh; repository không có spatial block/stability evaluation.

**Hậu quả:** plots/profiles chỉ giải thích clusters theo input, không chứng minh external validity. Random-row validation sẽ còn lạc quan do spatial dependence.

**Bắt buộc cho Q1 claim:** gọi đây là descriptive characterization. External validation phải dùng biến không tham gia clustering: held-out monitor/time period, future outcome, expert labels hoặc independent domain indicator. Air evaluation cần spatial blocking; Tiki nên dùng future revenue/retention hoặc business outcome ngoài 7 input features nếu có.

### H17. Runtime giữa methods không cùng measurement contract

**Loại:** [FAIRNESS]

- K-means baseline đo chỉ `fit_predict`: `scripts/run_public_kmeans_baseline.py:281`, `scripts/run_public_kmeans_baseline.py:283`.
- DEKM/MvDEC clock bao gồm shuffle, pretraining, refinement, repeated metrics và artifact I/O: `src/representation_learning/DEKM_dense.py:205`, `src/representation_learning/DEKM_dense.py:265`, `src/representation_learning/MVDEC_dense.py:1426`, `src/representation_learning/MVDEC_dense.py:1515`.
- FP-Max/FFS CSV bỏ runtime dù internal result có fit time.

**Hậu quả:** không được xếp hạng efficiency bằng các cột hiện tại.

**Bắt buộc:** log stage times riêng: load/preprocess, pretrain, refinement, FP-Max, FFS, final clustering, evaluation, serialization; báo wall-clock, peak CPU RAM/GPU RAM và hardware. Chỉ so cùng stage contract.

## Findings - Medium

### M1. Eigen direction của MvDEC 2025 tự mâu thuẫn; default phải được đặt tên là assumption, không phải truth

**Loại:** [FAIRNESS]

- DEKM 2021 p.4-5: eigenvalues tăng dần; eigenvalue nhỏ quan trọng hơn; chiều cuối/largest eigenvalue ít cluster structure nhất; `L4` kéo chiều cuối.
- MvDEC 2025 p.5 review lại logic này, nhưng p.8 gọi least-informative direction là smallest eigenvalue rồi vẫn nói thay “last dimension”.
- Code default `largest`, expose cả hai: `src/representation_learning/MVDEC_dense.py:53`, `src/representation_learning/MVDEC_dense.py:133`.

**Đánh giá:** expose cả hai là đúng hướng. Tuy nhiên `largest` phải gọi `DEKM-consistent`, `smallest` gọi `MvDEC-2025-literal-sentence`; preregister primary trước khi chạy. Không chọn direction dựa trên kết quả cuối.

### M2. FP-Max không lưu bin edges/effective bins; global warning suppression che collapsed bins

**Loại:** [BUG METADATA - BẮT BUỘC SỬA TRƯỚC FINAL RUN]

- `KBinsDiscretizer` được fit nhưng bin edges/effective bin count không được trả/lưu: `src/pipeline/fpmax.py:67`, `src/pipeline/fpmax.py:72`.
- Itemset table bỏ `feature_name` và grid CSV chỉ giữ selected names/min_support: `src/pipeline/fpmax.py:145`, `src/pipeline/fpmax.py:162`.
- CLI suppress toàn bộ warnings: `src/cli.py:283`; warnings về constant/collapsed bins có thể biến mất trong khi result vẫn ghi nominal `n_bins`.

**Hậu quả:** cùng `n_bins=7` có thể thực tế dùng ít bins khác nhau theo feature/seed mà artifact không cho audit.

### M3. K-means baseline chạy hai init strategies nhưng không có primary selection rule

**Loại:** [FAIRNESS]

- Grid gồm `k-means++` và `random`: `scripts/run_public_kmeans_baseline.py:37`, `scripts/run_public_kmeans_baseline.py:484`.
- Workbook aggregate theo từng init, nhưng manuscript có thể hậu nghiệm lấy dòng tốt hơn.

**Khuyến nghị:** preregister `k-means++` primary với `n_init=100`; `random` chỉ sensitivity. Không chọn init bằng ACC/NMI.

### M4. Runbook chứa commands bị code từ chối và parallelism có thể bùng nổ

**Loại:** [BUG DOCS - BẮT BUỘC SỬA]

- Runbook đặt đồng thời `candidate_workers=10` và `param_workers=5`: `RUNBOOK_GPU_TO_LOCAL.md:105`, `RUNBOOK_GPU_TO_LOCAL.md:106`.
- Code cấm cả hai lớn hơn 1: `src/pipeline/experiments.py:3537`.
- Outer workers kết hợp inner candidate workers tạo nested process pools và nhân bản dense matrices trên Windows.

### M5. Log append không có run identity và một số tracked log không còn là clean tabular artifact

**Loại:** [BUG ARTIFACT - BẮT BUỘC SỬA TRƯỚC FINAL RUN]

- `log_csv` luôn append theo filename: `src/representation_learning/utils.py:166`.
- MvDEC log name không chứa seed: `src/representation_learning/MVDEC_dense.py:846`.
- `log_history/REUTERS.csv` chứa nhiều blocks cùng seeds nhưng khác results, không có method/config/hash để phân biệt. `log_history/TIKI.csv` trộn schema lịch sử.

**Khuyến nghị:** một structured JSONL/CSV per run_id, schema version cố định; summary không parse append logs.

### M6. Tests pass nhưng không cover các failure quan trọng của release workflow

**Loại:** [BUG TEST COVERAGE - BẮT BUỘC BỔ SUNG KHI SỬA]

- `uv run pytest -q`: 103 tests pass.
- Tuy nhiên không có end-to-end test load tất cả tracked current artifacts; nếu có, Air artifact sẽ fail.
- Không có test default objective = intended protocol, public ablation path collision, private multi-run persistence, canonical dataset sample/hash, DEKM max-iter final recompute, no-ground-truth-in-train, singleton-before-selection hoặc memory guard.
- Public data tests dùng tiny synthetic files nên không bắt sai sample count: `tests/test_public_data.py:9`, `tests/test_public_data.py:30`.

### M7. Repository-wide formatter/linter chưa sạch

**Loại:** [TÙY CHỌN TRƯỚC SUBMISSION, BẮT BUỘC NẾU CI CLAIM]

- `uv run ruff check .` fail 66 lỗi, chủ yếu copied DEKM/external replication code.
- First-party pipeline lint scope pass, nhưng format check vẫn muốn sửa `scripts/fetch_public_datasets.py`, `src/pipeline/io.py` và hai test files.
- Nên exclude vendored release rõ ràng hoặc đưa nó vào archival subtree; không để global quality command fail mà README không nói.

### M8. Paper/code preprocessing correspondence cần được ghi là assumption

**Loại:** [FAIRNESS / REPORTING]

- DEKM 2021 p.6 mô tả text normalization sao cho `(1/d)||x_i||^2` xấp xỉ 1; release code và repository normalize mỗi row về unit L2 norm: `src/pipeline/public_data.py:38`.
- Air paper không publish scaler; repository chọn per-column Min-Max dựa vào Table 4: `src/representation_learning/MVDEC_dense.py:316`, `src/representation_learning/MVDEC_dense.py:321`.
- Air paper Table 3 liệt kê 17 features, p.13-14 lại nói/hiển thị 13. Repository chọn schema 13 từ Table 4.

**Đánh giá:** đây là các reproduction assumptions hợp lý, nhưng phải xuất hiện trong method/limitations và manifest; không gọi là exact preprocessing của paper.

## Findings - Low

### L1. Representation scripts dùng mutable module globals và nhiều hardcoded research values

**Loại:** [TÙY CHỌN, NHƯNG NÊN SỬA TRƯỚC MỞ RỘNG]

- Dataset architecture, K, paths, tolerances và loss weights sống ở module globals: `src/representation_learning/MVDEC_dense.py:28`, `src/representation_learning/MVDEC_dense.py:63`.
- Điều này làm function behavior phụ thuộc state do `__main__` mutate và khó test/parallelize.

**Khuyến nghị:** typed immutable run config được truyền vào model/train/artifact builder; không cần refactor lớn hơn phạm vi này.

### L2. Pickle được dùng làm artifact chính

**Loại:** [TÙY CHỌN / SECURITY-PORTABILITY]

- Load dùng `pickle.load`: `src/pipeline/data.py:310`, `src/pipeline/native_evaluation.py:138`.
- README đã giới hạn trusted local artifacts, nên không phải remote-code vulnerability trong current intended flow. Tuy nhiên pickle phụ thuộc Python/library và không an toàn khi tải từ release bên ngoài.

**Khuyến nghị:** manifest JSON + arrays NPZ/Zarr/Parquet; nếu giữ pickle, lưu hash, schema, Python version và chỉ load sau integrity check.

### L3. Diagnostics/logging của representation code chưa theo abstraction chung

**Loại:** [TÙY CHỌN]

- Representation scripts dùng `print()` và ad-hoc CSV trong khi phần pipeline dùng Loguru. Đây không làm sai số liệu nhưng làm logs khó structured/audit.

## Đối chiếu tóm tắt với hai paper

| Thành phần | DEKM 2021 | MvDEC 2025 | Repository hiện tại | Đánh giá |
|---|---|---|---|---|
| Text architecture | `d-500-500-2000-10`, decoder đối xứng | View 1 DEKM-like; public details không đầy đủ | View 1 khớp; View 2 U-Net dense | Hợp lý nhưng MvDEC public/view semantics là assumption |
| Pretraining | L1 reconstruction, sau đó bỏ decoder/L1 | Giữ L1 trong joint loss | DEKM bỏ L1; MvDEC giữ L1 | Phần này đúng theo khác biệt hai paper |
| L2/L3/L4 | L2/L3 giải thích geometry; optimization cuối dùng greedy L4 | Eq.11 dùng L1+L2+L3+L4 | Default MvDEC L1+L4 | Sai identity nếu gọi faithful MvDEC |
| Eigen | ascending; last/largest ít thông tin nhất | Nội bộ mâu thuẫn largest vs smallest | Default largest, expose cả hai | Phải đặt tên/preregister hai interpretations |
| Target | hard K-means; `y'` thay tọa độ cuối; không Student-t | Tương tự, không target distribution riêng | `selected_dimension_only` gần literal; `frozen_snapshot` là implementation ablation | Không gọi target mode là paper-defined parameter |
| Stop | `<0.1%` assignment changes hoặc Iter | chỉ “until convergence” | public 0.001; private 0.01 | Public khớp DEKM; private là assumption cần sensitivity |
| Runs | 3-run average; không std/seeds | “best performance”; không seeds/std | có mean/std public nếu explicit seed; private artifact overwrite | Chưa đạt full-pipeline reproducibility |
| Public data | Reuters 10k, 20NEWS 18,846, RCV1 10k | Table 1 mâu thuẫn 210/2,000/10k | local release Reuters 10k, 20NEWS 18,057; RCV1 cache external | Không được claim exact same dataset nếu chưa freeze hashes |
| Air data | không áp dụng | 3,765 rows; paper mâu thuẫn 13 vs 17 features | dùng 13-feature `data_demvk.csv`, Min-Max assumption | Chỉ là partial processed-data reproduction |
| Air Silhouette | không áp dụng | metric/space không nêu; K=4 visual elbow | Euclidean, numeric Gower, mixed Gower ở nhiều spaces | Không thể trực tiếp đối chiếu 61.76% nếu không biết paper space |

## Những phần đang làm đúng

- ACC dùng Hungarian assignment và NMI permutation-invariant đúng: `src/pipeline/external_metrics.py:32`; tests tương ứng pass.
- Canonical row restoration cho shuffled public runs hợp lý: `src/pipeline/public_artifacts.py:15`.
- MvDEC recompute final embeddings/K-means sau training, tránh stale state mà DEKM hiện mắc: `src/representation_learning/MVDEC_dense.py:1177`.
- Public tolerance 0.001, text latent architecture và K theo paper benchmark được encode rõ.
- Asymmetric binary Gower implementation có contract rõ và tests cho shared `0-0`: `src/pipeline/clustering.py:143`, `tests/test_clustering_diagnostics.py:42`.
- Manifest hiện đã chặn trộn data/representation hash ở dataset-scoped output; đây là nền tốt để mở rộng provenance.
- Tiki raw workbook tái tạo đúng canonical preprocessed matrix trong regression test; DBSCAN exclusions/mapping được kiểm tra: `tests/test_data_preprocessing.py:35`.

## Flow đánh giá tối ưu - Public datasets

### 1. Freeze dataset contract trước khi chạy model

1. Chọn và đặt tên một contract chính, ví dụ `dekm_release_v1`, gồm exact raw-file hashes, processed-array hash, expected shape, ordered sample IDs, labels hash, K và preprocessing.
2. Với local files hiện tại, ghi rõ Reuters 10,000 và 20NEWS 18,057. Không gọi 20NEWS 18,846 nếu array thực tế không có số rows đó.
3. Materialize RCV1 X/y, selected row indices và 2,000 feature indices; không phụ thuộc global scikit-learn cache ở runtime.
4. MvDEC 2025 Table 1 chỉ dùng như published-reference vì sampling protocol mâu thuẫn/không đủ. Không claim exact replication Table 1.

### 2. Tách faithful reproduction và controlled comparison

- **DEKM-release-faithful:** giữ behavior release để đối chiếu published code, kể cả update interval; report rõ environment legacy.
- **DEKM-controlled:** fixed `n_init`, final recompute, same full-epoch/step budget và stop metadata.
- **MvDEC-2025-literal:** explicit Eq.11-equivalent objective, explicit eigen interpretation, same dataset contract.
- **MvDEC-DEKM-consistent:** largest-eigen interpretation vì phù hợp DEKM mathematics.
- Current `L1+L4` phải là ablation, không là default unnamed MvDEC.

### 3. Full-pipeline paired seeds

- Khuyến nghị **20 seeds** cố định và versioned; tối thiểu 10 nếu GPU budget hạn chế.
- Cùng một seed phải điều khiển Python/NumPy/TensorFlow/Keras shuffle, initialization, K-means và downstream clustering.
- Mỗi seed chạy độc lập K-means, DEKM, MvDEC, MiMvDEC without FFS và with FFS; không lấy “best seed”.
- Failed run được giữ trong denominator và báo nguyên nhân. Chỉ retry cùng seed cho system failure; không thay bằng seed khác.

### 4. Ground truth protocol

- Train/mining/selection process không được mount/read labels.
- Config và assignments phải được freeze/hashes ký trước final evaluator.
- Direct paper replication có thể dùng class-count K nhưng phải gọi **oracle-K**.
- Nếu manuscript claim fully unsupervised, thêm **estimated-K** protocol label-blind và không dùng class count cho selection.

### 5. Metrics public

- **Primary/co-primary:** ACC và NMI, báo mean +/- sample std qua full-pipeline paired seeds.
- **Secondary:** ARI, failed-run rate, cluster-size distribution.
- **Diagnostic only:** native Euclidean/Gower Silhouette; không dùng để xếp hạng cross-space.
- Lưu row-ordered assignments cho mọi method/seed. ACC alignment chỉ thực hiện tại final evaluation; seed-stability dùng ARI/NMI giữa assignments, không align bằng ground truth.

### 6. Statistics public

- Planned contrasts tối thiểu: DEKM vs MvDEC; MvDEC vs MiMvDEC-noFFS; MiMvDEC-noFFS vs MiMvDEC-FFS.
- Dùng paired exact permutation test hoặc Wilcoxon signed-rank trên per-seed deltas; báo median/mean delta và rank-biserial hoặc paired effect size.
- Báo 95% bootstrap CI của paired delta.
- Holm correction qua planned contrasts/datasets/primary metrics; không test mọi grid row.
- Chỉ dùng Friedman/Iman-Davenport khi có đủ nhiều datasets; với 3 datasets hiện tại, report dataset-specific paired effects quan trọng hơn một omnibus rank test yếu.

## Flow đánh giá tối ưu - Tiki và Air Pollution

### 1. Vai trò đúng của ba evaluation spaces

| Space | Mục đích đúng | Có dùng làm primary cross-method không? |
|---|---|---|
| Native | Model có tạo geometry tốt trong representation riêng hay không | Không; chỉ diagnostic, vì feature space/dimension khác nhau |
| Controlled mixed-space | So assignments trên đúng cùng frozen augmented matrix/distance; cô lập reclustering/FFS trong một contrast | Có cho contrast cục bộ, nếu matrix giống hệt giữa hai arms |
| Common original-space | So mọi assignments trên cùng ordered original numeric matrix | Có; đây là direct private-dataset comparison chính, nhưng phải freeze scaling |

Không so magnitude Euclidean với Gower. Không so two native-Gower scores như thể cùng thang đo chỉ vì cùng tên metric.

### 2. Private metrics

- **Primary:** common-space numeric Gower Silhouette trên cùng frozen data matrix.
- **Co-primary:** clustering stability qua repeated outer resampling; với Air phải là spatial-block stability, không random-row stability.
- **Secondary:** standardized-common-space Euclidean Silhouette, negative-silhouette fraction, per-sample Silhouette std, cluster-size balance.
- **Controlled diagnostic:** asymmetric và symmetric mixed-Gower trên cùng frozen FP-Max matrix.
- **Native diagnostic:** Euclidean/Gower trong learned representation để đối chiếu paper/geometry, không dùng làm decisive ranking.
- Nếu có external outcomes, dùng chúng làm external validation và giữ ngoài clustering features.

### 3. Selection/evaluation split

- **Tiki:** repeated outer splits hoặc cross-fitting. Fit scaler/winsorization, MvDEC, bin edges, FP-Max itemsets và FFS chỉ trên discovery fold; freeze encoder/itemsets/prototypes rồi assign validation rows.
- **Air Pollution:** dùng spatial blocks; nếu còn temporal raw data, ưu tiên temporal/spatial holdout. Random split không đủ vì neighbor-NO2 và land-use spatial autocorrelation.
- Khuyến nghị 20 outer seeds/splits; inner selection dùng repeated subsampling/stability.
- K, strategy, bins, support, max features và stopping margin được chọn inner-only rồi freeze.

### 4. Feature selection protocol

1. Fit binning trên discovery data; persist exact edges, effective bins và constant/collapsed-bin status.
2. Mine FP-Max chỉ trên discovery data; persist all maximal itemsets, support, item order và feature hashes.
3. Apply frozen itemsets sang validation/test; không re-mine trên test.
4. FFS objective nên kết hợp held-out Silhouette/stability và complexity penalty hoặc one-standard-error rule.
5. Stop khi không candidate nào có predeclared robust improvement, ví dụ lower CI > 0 hoặc margin domain-defined; không dùng `1e-6`.
6. Final outer evaluation chạy đúng một lần; grid/full trace vẫn được lưu nhưng không dùng để chọn sau khi xem result.

## Ablation tối thiểu để cô lập contribution

Trong mỗi seed, giữ cùng MvDEC artifact, K, downstream backend, restart budget và evaluation contracts:

| Arm | Representation/backend | Contribution được cô lập |
|---|---|---|
| A0 | K-means trên original preprocessed X | Shallow baseline |
| A1 | DEKM faithful/controlled | DEKM representation |
| A2 | MvDEC + original K-means | Multi-view representation/loss |
| A3 | `h_fused` + downstream backend, không binary feature | Backend/reclustering effect |
| A4 | `h_fused` + individual one-hot bins, không FP-Max itemsets | Discretization/binary expansion effect |
| A5 | `h_fused` + full FP-Max set, không FFS | FP-Max beyond simple bins |
| A6 | Cùng candidate set A5 + FFS subset | FFS contribution |

Các ablation eigen direction, target mode, loss weights và stopping tolerance phải là factorial study riêng. Không dùng artifacts khác seed/loss/batching để so target mode.

## Artifact và manifest cần có

Mỗi run manifest tối thiểu phải lưu:

- `run_id`, schema version, git commit, dirty diff hash, timestamp.
- Dataset contract/version, raw/processed hashes, ordered sample IDs/hash, feature order, view mapping.
- Full environment: Python, OS, dependency lock hash, TensorFlow/CUDA/cuDNN/GPU/CPU.
- Full seed map và deterministic flags.
- K, architecture, optimizer/lr, pretrain/refinement budgets, K-means settings.
- L1/L2/L3/L4 definitions/weights, eigen direction/order/index, target mode, tolerance, stop reason.
- Assignments, assignment hash, cluster sizes, representation hash/path.
- FP-Max bin edges/effective bins, all itemsets/supports, generated feature order/hash.
- FFS candidate trace, selected features/order, stopping reason/margin.
- Stage runtimes, peak RAM/GPU RAM, final status/error.
- Final metrics chỉ được sinh bởi evaluator xác nhận exact dataset hash, protocol và paired seed set.

CSV summary vẫn hữu ích, nhưng authoritative artifact nên là manifest JSON + arrays NPZ/Zarr/Parquet. Không dựa vào append logs để dựng bảng paper.

## Thay đổi code tối thiểu trước khi chạy lại paper

### P0 - Bắt buộc trước khi dùng kết quả cho submission

1. **Freeze algorithm protocols:** sửa objective naming/weights của MvDEC; có explicit literal/equivalent/hybrid modes.
2. **Unique run outputs:** encode seed + full config hash; không overwrite; public ablations cũng có tag.
3. **Regenerate artifacts:** thay Air legacy; regenerate Tiki selected/frozen bằng cùng code; strict schema validation cho mọi dataset.
4. **Fix DEKM:** fixed `n_init`, final recompute, stop reason/max-iter status, seed/config persistence.
5. **Remove labels from train:** external metrics chỉ chạy sau artifact freeze.
6. **Freeze dataset contracts:** expected shapes/hashes/sample IDs; materialize RCV1; cấm supervised processed mode khỏi main table.
7. **Implement full-pipeline seed orchestration:** paired per-seed manifests và aggregate mean/std/deltas.
8. **Make FP-Max/FFS split-aware:** frozen bins/itemsets, held-out selection/evaluation, robust stopping.
9. **Fix ablation/backend contract:** thêm no-FP and one-hot-only arms; same backend/config for FP-Max/FFS contrasts.
10. **Fix singleton selection:** reject invalid candidates before argmax.
11. **Make distance scalable:** no full dense Gower for full 20NEWS; memory guard/chunked/subsample contract.
12. **Expand manifest:** full config, selected features/itemsets, hashes, runtime, environment; validate sample IDs.
13. **Pin environments/determinism:** separate release-faithful and modern-controlled locks.

### P1 - Nên hoàn tất trước camera-ready

1. Chuẩn hóa common Euclidean matrix; spatial-block evaluation cho Air.
2. Structured per-run logs; sửa runbook worker commands.
3. Integration tests cho tracked artifacts, CLI end-to-end, output collision, full seed pairing và canonical dataset hashes.
4. Persist fit/predict prototypes để evaluation rows thực sự out-of-sample.
5. Report resource usage và failure policy.

### P2 - Cải tiến tùy chọn

1. Chuyển pickle sang portable array artifacts.
2. Thêm CI/pre-commit, vulnerability audit và repository-wide format policy có exclude vendored code rõ ràng.
3. Refactor representation globals thành typed config sau khi protocol đã freeze; không cần redesign toàn pipeline.

## Kết luận review

Repository có nhiều bước engineering tốt: row alignment, external metrics, asymmetric-Gower contract, atomic CSV writes, artifact hash guard và bộ unit tests khá rộng. Tuy nhiên, trạng thái hiện tại **chưa đủ để hỗ trợ kết luận Q1** rằng MvDEC tốt hơn DEKM, FP-Max cải thiện MvDEC, hoặc FFS tạo gain độc lập. Các blocker lớn nhất là sai/không ổn định identity MvDEC, artifact overwrite/stale, selection-on-evaluation-data, không có full-pipeline paired seeds, ablation confounded và exact Gower không scale cho public data.

Không nên sử dụng các artifact/results hiện tại làm final paper tables trước khi hoàn tất P0 và chạy lại toàn bộ protocol đã freeze.
