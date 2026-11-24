# Kế hoạch: paper "FP-Max làm tốt hơn phân cụm" dựa trên DEKM, MvDEC và Intuitive

## Bối cảnh

Mục tiêu của bạn là một paper với điểm nhấn là FP-Max: thêm đặc trưng "itemset tối đa phổ biến" vào các thuật toán phân cụm (truyền thống, Intuitive, DEKM, MvDEC) để chúng phân cụm tốt hơn. Có Forward Feature Selection (FFS) đi kèm, và Tiki cùng AIRPOLLUTION là phần ứng dụng.

Tôi đã đọc toàn bộ 3 paper và audit toàn bộ codebase (3 luồng đọc song song, mỗi file đọc trọn vẹn).

### Hiện trạng tái hiện

| Thuật toán | Trạng thái | Chi tiết |
|---|---|---|
| DEKM | Đã tái hiện | REUTERS, 20NEWS, RCV1, 3 seed, sai lệch so với paper trong khoảng seed. Ví dụ REUTERS: ACC 0.771 so với paper 0.763, NMI 0.580 so với 0.591. |
| Intuitive-K-prototypes | Tái hiện ở mức chấp nhận được | 8 bộ UCI. Iris và BCW khớp chính xác. Heart-5 có std cao hơn paper. Một số tham số phải tự fit (beta, mu, chebyshev cho Ionosphere). |
| MvDEC | Không tái hiện được | Trần NMI trên REUTERS khoảng 0.597, paper báo 0.614. Paper tự mâu thuẫn ở nhiều chỗ (Eq.4 lấy trung bình, Fig.2 lại nối hai view; Table 1 ghi 210 mẫu). Tác giả không trả lời sau 2 mail. |

### Trả lời câu hỏi "chỉ đánh giá bằng silhouette thôi được không?"

**Không nên.** Audit tìm ra đúng lỗi khiến silhouette-only bị phản biện bác:

1. **C1: hai bên chấm ở hai không gian khác nhau.** Baseline MvDEC được chấm bằng Gower số trên `h_fused` (`src/pipeline/data.py:1787`). Phương pháp có FP-Max được chấm bằng Gower hỗn hợp có chứa chính các cột FP-Max của nó (`src/pipeline/forward_selection.py:107-110`). Hai con số này không so sánh được với nhau.
2. **C2: chọn và báo cáo bằng cùng một con số.** FFS, chọn init, chọn tham số Intuitive và chọn dòng tốt nhất trong grid 171 cấu hình đều tối đa hóa silhouette, rồi báo cáo lại đúng silhouette đó.
   - Thêm một cột nhị phân chia đôi dữ liệu, rồi phân cụm theo cột đó, thì silhouette gần như tự động tăng. Test `tests/test_best_evaluation.py:62,84` cho thấy đúng điều này.
3. **C3: FP-Max không mang thông tin mới.** Itemset được rút ra từ chính các cột đã chia bin, nên phần tăng là do "đánh trọng số lại" dữ liệu cũ, không phải do có thêm dữ liệu. Paper phải nói thẳng điều này.
4. **C4: thiếu đối chứng.** Chưa có dòng "cùng thuật toán, không FP-Max". Hiện K-Prototypes hoặc Intuitive có FP-Max đang được so với K-Means của MvDEC, nên lẫn hai hiệu ứng: đổi thuật toán và thêm FP-Max.

**Cách đúng:**
- Silhouette vẫn được dùng làm **tiêu chí nội bộ** của FFS. Việc này hợp lệ vì không dùng nhãn.
- Bằng chứng chính lấy từ **bộ dữ liệu có nhãn**: ARI, NMI, ACC. Nhãn không bao giờ đi vào lúc chọn, nên không có vòng lặp.
- Silhouette chỉ báo cáo phụ, và chấm trong **một không gian tham chiếu cố định cho mọi phương pháp**: đặc trưng gốc đã scale, khoảng cách Gower. Kèm theo:
  - Davies-Bouldin và Calinski-Harabasz;
  - độ ổn định: ARI giữa các seed và các lần bootstrap.
- Baseline được dò tham số với **cùng số lần thử và cùng tiêu chí** như phương pháp có FP-Max.

### Các lỗi code khác cần sửa trước

- **C5: test đang chạy nhầm model.** Hai package cùng tên `intuitive_kprototypes`; pytest nạp bản trong `paper_replication`. Bản pipeline (`src/intuitive_kprototypes/model.py`) chưa có test thật nào, mọi test đều mock nó.
- **M1: hai model Intuitive đã lệch nhau.** Bản pipeline không có `mu_convention`; mu theo Eq.26 in trong paper làm sập cụm. Hai bản còn khác nhau ở công thức CV (`|mean|`) và ở cách khởi tạo farthest.
- **M2: Intuitive và K-Prototypes nhận `h_fused` chưa scale.** Độ lớn của embedding quyết định khối số hay khối nhị phân chiếm ưu thế.
- **M3, M5, M6, M7: lỗi im lặng.**
  - Cụm singleton chỉ bị loại sau khi đã chọn xong.
  - `except` nuốt mọi lỗi.
  - Cụm rỗng làm `best_evaluation` crash.
  - Các run lỗi biến mất khỏi bảng so sánh (survivor bias).
- **Chưa có đường DEKM + FP-Max.** `load_mvdec_result` bắt buộc phải có `h_fused`, còn DEKM lưu embedding dưới key `embedding`.
- **MvDEC có 3 protocol cùng được gọi là "canonical", và dùng biến toàn cục mutable.** Số chiều, k và tolerance là biến toàn cục (`MVDEC_dense.py:138-159, 3305`).
- **Schema checkpoint DEKM_F đã lệch.** Không import được checkpoint mới, nên không chạy được MvDEC nhiều seed với View1 import.
- **Rò rỉ nhãn.** Biến thể text trong `scripts/fetch_public_datasets.py:665-674` chọn đặc trưng bằng chi² theo nhãn thật. Tuyệt đối không dùng bản này.
- **Vấn đề dữ liệu.**
  - Tiki có 2 cột gần như hằng sau winsorize.
  - AIRPOLLUTION trộn nhiều đơn vị đo, và không dựng lại được từ nguồn gốc.

## Hướng đề xuất

Đây là phương án B từ trước, giờ đã cụ thể hóa.

### Thiết kế thí nghiệm (factorial)

**Backbone**, tức khối số đưa vào phân cụm:
- `raw`: đặc trưng gốc đã scale;
- `dekm`: embedding DEKM;
- `mvdec`: `h_fused`, chỉ để ở phụ lục và ghi rõ "chưa tái hiện".

**Thuật toán phân cụm:**
- K-Means: chỉ dùng cho khối số, không có FP-Max; đóng vai trò mốc tham chiếu;
- K-Prototypes;
- Intuitive-K-prototypes (protocol đã đóng băng).

**Mức FP-Max:**
- `none`: cùng thuật toán, khối nhị phân rỗng;
- `fpmax`: toàn bộ itemset;
- `fpmax+ffs`.

**Quy tắc tích hợp, giống hệt nhau ở mọi nơi:**
- FP-Max được khai thác trên **chính khối số mà thuật toán nhận**: `raw` thì khai thác trên đặc trưng gốc, deep thì khai thác trên embedding.
- Khối số được min-max scale về [0, 1] trước khi vào K-Prototypes và Intuitive.

**Cách chọn siêu tham số FP-Max (bins, strategy, min_support):**
- Chọn bằng silhouette, không dùng nhãn.
- Tách chọn và đánh giá: chọn trên seed backbone A, báo cáo trên các seed B..J.
- Báo cáo cả phân phối của toàn grid, không chỉ giá trị max.

### Dữ liệu

| Nhóm | Bộ dữ liệu | Máy | Vai trò |
|---|---|---|---|
| Bảng có nhãn (chính) | 8 bộ UCI của paper Intuitive: Iris, Ionosphere, BCW, Soybean, Australian, Heart-2, Heart-5, Chess | CPU | Kết quả chính |
| Bảng có nhãn (bổ sung) | Các bộ UCI số đã có trong `configs/public_datasets.json`: wine, wdbc, seeds, glass… | CPU | Bổ sung |
| Văn bản có nhãn | REUTERS-10K, RCV1-10K, 20NEWS (sklearn 18,846), dùng loader DEKM, không dùng bản chi² | GPU Kaggle | Backbone `dekm` (và `mvdec`) |
| Ứng dụng, không nhãn | Tiki, AIRPOLLUTION | CPU | Silhouette, DB và CH trong không gian tham chiếu, độ ổn định, diễn giải cụm. Không dùng để kết luận "FP-Max tốt hơn". |

### Thống kê

- Mỗi ô chạy 10 seed, báo cáo mean ± std.
- So sánh từng cặp "có FP-Max" với "không FP-Max" trong cùng backbone và cùng thuật toán.
- Kiểm định Wilcoxon signed-rank trên các cặp (bộ dữ liệu × seed), hiệu chỉnh Holm, kèm effect size.
- Báo cáo mọi ô, kể cả các ô FP-Max thua.

## Các bước

### Bước 0. Dọn nền

- Đổi tên package sandbox thành `intuitive_kprototypes_sandbox`.
- Gộp hai model Intuitive về một lõi duy nhất, có các option `mu_convention`, `init_strategy` và `view_weight_alpha`. Bản pipeline dùng protocol đã đóng băng trong `configs/intuitive_protocols.json`.
- Viết test thật, không mock, cho model của pipeline.
- Chốt **một** protocol MvDEC dùng cho FP-Max và ghi lý do.
- Sửa schema checkpoint DEKM_F.
- Xóa hoặc chặn biến thể chi² trong fetch script.

### Bước 1. Sửa đánh giá (C1, C2, C4, M3, M5, M6, M7)

- Module mới `src/pipeline/reference_evaluation.py`, gom một chỗ (DRY):
  - không gian tham chiếu cố định cho mỗi dataset;
  - silhouette, DB và CH;
  - ARI, NMI, ACC: tái sử dụng `src/pipeline/external_metrics.py`;
  - độ ổn định.
- Thêm dòng `none` cho K-Prototypes và Intuitive: gọi `run_kprototypes` và `run_intuitive_kprototypes` với `binary_df` rỗng, với cùng ngân sách dò tham số.
- Loại singleton và cụm rỗng ngay trong vòng chọn ứng viên. Giữ lại các run lỗi trong bảng tổng hợp.
- Đặt `min_improvement` của FFS là một ngưỡng có ý nghĩa, ví dụ 0.005, và ghi vào protocol.
- Test: baseline và phương pháp phải được chấm trong cùng một không gian; một test hồi quy cho trường hợp "cột trùng nhãn".

### Bước 2. Interface backbone chung

- `load_backbone(kind, artifact_path)` trả về một DataFrame số cho cả ba loại:
  - `raw`;
  - `dekm`: key `embedding` trong artifact DEKM;
  - `mvdec`: key `h_fused`.
- Nó thay cho việc `experiments.py` chỉ nhận MvDEC.
- Một runner factorial duy nhất (backbone × thuật toán × mức FP-Max × seed) thay cho 6 job runner đang trùng lặp. Tái sử dụng:
  - `prepare_fpmax_data` và `extract_fpmax_features_from_prepared` (`src/pipeline/fpmax.py`);
  - `run_forward_selection`.
- Kết quả ghi ra CSV dạng dài (long format), mỗi dòng một run.

### Bước 3. Chạy thí nghiệm

1. Bảng UCI, backbone `raw`, trên CPU. Đây là kết quả chính và có sớm nhất.
2. DEKM: chạy 10 seed embedding trên Kaggle cho 3 bộ văn bản và cho các bộ UCI. Mạng MLP nhỏ, có thể chạy CPU. Sau đó chạy factorial trên CPU.
   - Gower O(N²) trên 10k–18k mẫu sẽ tràn bộ nhớ. Không gian tham chiếu cho văn bản dùng cosine trên tf-idf, tính silhouette trên mẫu con cố định 3,000 điểm.
3. MvDEC với protocol đã chốt, đưa vào phụ lục.
4. Tiki và AIRPOLLUTION: phần ứng dụng.

### Bước 4. Tổng hợp

- Bảng chính: Δ ARI và Δ NMI (FP-Max trừ none) theo backbone × thuật toán, kèm p-value đã hiệu chỉnh Holm.
- Bảng phụ lục: mọi ô.
- Ghi rõ các deviation:
  - Intuitive dùng mu đảo (`1 − mu`);
  - MvDEC chưa tái hiện được;
  - FP-Max được khai thác từ chính các đặc trưng đầu vào.

## Rủi ro và cách nói thật trong paper

- **FP-Max có thể không giúp** trên ARI/NMI. Thiết kế này vẫn cho ra kết quả công bố được: nó cho biết FP-Max giúp ở đâu và vì sao, ví dụ trên backbone `raw` hay `dekm`, với K-Prototypes hay Intuitive. Không được chọn lọc kết quả.
- **MvDEC không tái hiện được**, nên chỉ để ở phụ lục.
- **Intuitive phụ thuộc tham số tự fit.** Dùng protocol đã đóng băng và nêu rõ trong paper.

## File chính

**Sửa:**
- `src/pipeline/experiments.py`
- `src/pipeline/forward_selection.py`
- `src/pipeline/clustering.py`
- `src/pipeline/data.py` (`load_mvdec_result`)
- `src/pipeline/best_evaluation.py`
- `src/pipeline/common_evaluation.py`
- `src/intuitive_kprototypes/model.py`
- `paper_replication/intuitive_kprototypes/src/`
- `src/pipeline/dekm_checkpoint.py`
- `scripts/fetch_public_datasets.py`
- `src/cli.py`

**Mới:**
- `src/pipeline/reference_evaluation.py`
- `src/pipeline/backbones.py`
- `configs/fpmax_study.json` (pre-register các ô thí nghiệm, seed, grid)

**Tái sử dụng:**
- `src/pipeline/external_metrics.py`
- `fpmax.py`
- `run_kprototypes` và `run_intuitive_kprototypes`
- `paper_replication/.../run_paper_benchmark.py` (protocol tuned)

## Kiểm tra

- `uv run --python 3.12 python -m pytest tests paper_replication/intuitive_kprototypes/tests -q`: mọi test pass, kể cả các test mới (cùng không gian chấm, dòng `none`, loader DEKM, model Intuitive không mock).
- `uv run --python 3.12 ruff check src tests`.
- Smoke test: chạy factorial trên Iris và BCW với 2 seed và kiểm tra:
  - dòng `none` của K-Prototypes khớp với kết quả K-Prototypes trong paper Intuitive (bảng baseline);
  - Intuitive `none` khớp với số đã tái hiện.
- Kiểm tra chéo: mỗi ô của CSV kết quả có đủ 10 seed, và các run lỗi được đếm chứ không bị bỏ.
