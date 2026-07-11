# Kế hoạch: Implement paper 2025 Multi-view DEKM (MVDEC) cho dữ liệu ô nhiễm không khí

## Bối cảnh

Dự án đang xây `src/representation_learning/MVDEC_dense.py` — bản mở rộng
2-view của `DEKM_dense.py` (paper DEKM 2021), theo đúng paper *Multi-view Deep
Embedded Clustering: Exploring a new dimension of air pollution* (2025). Mục
tiêu cuối: một script chạy được, sinh ra embedding đã fuse (2 view) và cụm
cho dataset ô nhiễm không khí Luxembourg (`data/preprocessed_data/data_demvk.csv`,
3765 dòng × 13 đặc trưng, không có nhãn thật).

**Đã xác nhận/chốt trong các lượt trước (không cần bàn lại):**
- `n_clusters = 4` (paper tự tìm bằng Elbow method cho đúng dataset này).
- `hidden_units = 10` — số chiều embedding, **bắt buộc bằng nhau** giữa 2 view
  vì Eq. 4 fuse bằng trung bình cộng trực tiếp (`h = (h1+h2)/2`).
- `input_shape = 13`.
- Không có nhãn thật → dùng `silhouette_score` thay ACC/NMI, không dùng
  `utils.py::get_ACC_NMI`.
- Style code: bám sát `external_repos/DEKM/DEKM_dense.py` — hàm trần trụi,
  biến cấu hình global, không dùng class/dataclass, ưu tiên **giữ y nguyên cơ
  chế gốc** (Hungarian-matching, điều kiện dừng <0.1% đổi cụm, mini-batch
  index liên tục) hơn là viết lại "sạch" hơn.
- **Kiến trúc 2 view đã viết xong và đã build-test** (`model_view1`,
  `model_view2` trong `MVDEC_dense.py`, output `(None,23)` cả 2, khớp Fig. 2
  của paper) — đây là 85 dòng hiện có trong file, KHÔNG viết lại.

**Quy tắc bắt buộc cho phiên làm việc này** (rút kinh nghiệm từ việc đã lỡ
code vượt phạm vi 1 lần): làm **đúng 1 bước** trong danh sách dưới đây mỗi lần,
dừng lại sau khi viết xong + kiểm tra nhẹ (build/shape-check, KHÔNG chạy
training đầy đủ), rồi chờ xác nhận trước khi sang bước kế tiếp. Không tự ý
gộp nhiều bước lại dù có vẻ tiện.

## Danh sách các bước (theo đúng thứ tự Eq. 3 → Eq. 11 của paper)

**Bước 1 — Kiến trúc 2 view.** ✅ Đã xong (`model_view1`, `model_view2`).

**Bước 2 — Đọc dữ liệu.** Hàm `get_xy(ds_name='AIRPOLLUTION', ...)`: đọc
`data/preprocessed_data/data_demvk.csv` bằng pandas, ép kiểu `float32`,
shuffle theo đúng cách `DEKM_dense.py::get_xy` đang làm (dùng `tf.random.shuffle`
với `shuffle_seed`). Không có `y` (trả về 1 mình `x`). Kiểm tra nhẹ: gọi hàm,
in ra `x.shape` phải là `(3765, 13)`.

**Bước 3 — Pretrain riêng từng view (Eq. 5).** `loss_train_base` (dùng chung
cho cả 2 view, y hệt bản gốc: MSE giữa input và phần reconstruction
`y_pred[:, hidden_units:]`), `train_base_view1(ds_xx)`, `train_base_view2(ds_xx)`
(mỗi hàm: build model tương ứng, compile, fit, lưu trọng số riêng
`weight_base_view{1,2}_{ds_name}.weights.h5`). Kiểm tra nhẹ: pretrain vài
epoch (không cần đủ 200), xem loss có giảm dần qua các epoch không (không cần
hội tụ hẳn).

**Bước 4 — Fuse 2 embedding (Eq. 4).** Không cần hàm riêng — chỉ là
`H = (h1 + h2) / 2` ngay trong `train()` ở bước 6, nhưng kiểm tra ở bước này
trước khi ghép vào vòng lặp lớn: build 2 model đã pretrain (bước 3), chạy
forward trên vài dòng dữ liệu thật, lấy `[:, :hidden_units]` mỗi bên, cộng
trung bình, in shape phải là `(n, 10)`.

**Bước 5 — K-means + eigen-decompose trên `H` đã fuse (Eq. 6–9).**
`sorted_eig(X)` (copy y hệt bản gốc). Kiểm tra nhẹ: dùng `H` đã fuse ở bước 4,
chạy 1 lần `KMeans(n_clusters=4).fit(H)`, tính within-class scatter `S`
(y hệt cách `DEKM_dense.py::train` làm — vòng `for i in range(n_clusters)`
cộng dồn `(H[assignment==i]-U[i])^T @ (...)`), gọi `sorted_eig(S)`, in ra
`silhouette_score(H, labels)` một lần — không cần đúng số cao, chỉ cần chạy
không lỗi.

**Bước 6 — Vòng lặp huấn luyện chính `train()` (Eq. 10–11).** Ghép toàn bộ
bước 4+5 vào đúng cấu trúc vòng lặp của `DEKM_dense.py::train`: giữ nguyên
Hungarian-matching, điều kiện dừng `n_change_assignment <= len(x)*0.001`,
mini-batch index liên tục. Khác biệt bắt buộc so với bản gốc: `H` là fused
(bước 4), bước gradient dùng `tf.GradientTape()` cập nhật
`model1.trainable_variables + model2.trainable_variables` cùng lúc (1
optimizer Adam chung), trả về `silhouette` thay vì `(acc, nmi)`. Kiểm tra
nhẹ: chạy 2-3 vòng lặp ngoài (không cần chạy tới khi hội tụ hẳn) bằng model
**chưa pretrain** để xác nhận không lỗi shape/dtype ở bước gradient — đã làm
việc này 1 lần rồi (xem lịch sử hội thoại), chỉ cần lặp lại nhanh sau khi
viết chính thức vào file.

**Bước 7 — Khối `__main__`.** argparse (`ds_name`, `--runs`, `--seed`, y hệt
`DEKM_dense.py`), set các global (`input_shape=13, n_clusters=4,
hidden_units=10, pretrain_epochs=200, batch_size=256, update_interval=10`),
vòng lặp `--runs` lần: `get_xy` → pretrain cả 2 view → `train()` → log qua
`utils.py::log_csv` (tái dùng, không viết lại) → in trung bình silhouette
cuối cùng. Kiểm tra nhẹ: chỉ chạy cú pháp (`python -m py_compile`), chưa chạy
thật.

**Bước 8 — Chạy thật lần đầu (chỉ khi được yêu cầu).** Chạy toàn bộ script
với cấu hình đầy đủ trên `data_demvk.csv`, xem silhouette hội tụ về đâu, đối
chiếu với silhouette của DEKM 1-view (baseline) để xem multi-view có tốt hơn
không — đúng tinh thần paper tự so sánh MvDEC với DEKM.

**Bước 9 — Đối chiếu benchmark với DEKM gốc trên REUTERS/20NEWS/RCV1.**
Paper 2025 tự so sánh MVDEC với DEKM trên 3 bộ text này (đã có sẵn dữ liệu
local tại `external_repos/DEKM/datasets/{REUTERS,20NEWS,RCV1}`, gitignored).
Khác biệt quan trọng so với air pollution: 3 bộ này **có nhãn thật**, nên
paper/DEKM gốc dùng ACC+NMI để đánh giá, không phải silhouette. Cần:

- Thêm nhánh cấu hình dataset trong `__main__` (`REUTERS`: `input_shape=2000,
  n_clusters=4`; `20NEWS`: `n_clusters=20`; `RCV1`: `n_clusters=4`;
  `hidden_units=10` cho cả 3, theo đúng quy ước gốc của DEKM cho dữ liệu
  văn bản, khớp cả 2 view).
- Tái dùng `get_xy` **đã có sẵn** trong `utils.py` (đã hỗ trợ REUTERS/20NEWS/
  RCV1) — import trực tiếp, không viết lại logic đọc dữ liệu.
- `train()` cần hỗ trợ thêm nhánh tính ACC/NMI khi có `y` (tái dùng
  `utils.py::get_ACC_NMI`), giữ nguyên nhánh silhouette-only cho trường hợp
  không có nhãn (air pollution) — quyết định cách hỗ trợ cả 2 (thêm tham số
  `y=None` hay viết hàm riêng) để lúc làm đến bước này mới chốt.
- Baseline đối chiếu: chạy `DEKM.py`/`DEKM_dense.py` gốc (pristine clone,
  1-view) trên cùng 3 bộ để có ACC/NMI làm mốc so sánh với MVDEC 2-view.

## File duy nhất bị sửa

`src/representation_learning/MVDEC_dense.py` — không đụng
`DEKM.py`/`DEKM_dense.py`/`utils.py` (giữ nguyên là bản clone y hệt
`external_repos/DEKM/`, đã xác nhận checksum khớp).

## Kiểm chứng cuối cùng (sau bước 8)

- `uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION --runs 3`
  chạy từ thư mục gốc, không crash.
- Silhouette cuối cùng được log vào `log_history/AIRPOLLUTION.csv` qua
  `utils.py::log_csv` có sẵn.
