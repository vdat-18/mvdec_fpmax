# REUTERS MvDEC architecture diagnosis

## Mục tiêu

Tài liệu này ghi lại ablation có kiểm soát nhằm kiểm tra giả thuyết: kết quả
REUTERS thấp hơn MvDEC 2025 có phải do vị trí latent của View2 hay không. Đây là
một chẩn đoán kiến trúc trên một seed, không phải bằng chứng thống kê cuối cùng
và không thay thế protocol public multi-seed.

Nguồn đối chiếu chính:

- `docs/2025_Multi-view Deep Embedded Clustering- Exploring a new dimension of air pollution.pdf`
- Fig. 2, p.6; Eq. (3), p.7; Eq. (4)-(6), p.8; Table 2, p.9.

## Provenance và contract thí nghiệm

Run được thực hiện trên Kaggle bằng Tesla P100 16 GB, TensorFlow 2.18.0, seed
42, commit `3b8cc0aaf3a51ee66fbb43818accdc544fa43e2c`. Dataset là release array
REUTERS `10,000 x 2,000`, `K=4`, với source SHA-256:

```text
427993db6b0544dfce1b8e3985e63e248d4d006de122cb202c504489d951706b
```

Hai arm dùng cùng dataset, row order, seed, preprocessing, View1, loss,
K-means schedule, stopping tolerance và final evaluation. Protocol contract xác
nhận `schedule_equal=True` và `objective_equal=True`. Khác biệt chủ đích duy
nhất về thuật toán là View2 architecture:

| Arm | Protocol | Config hash | View2 architecture |
|---|---|---|---|
| A0 | `mvdec_2025_public_reproduction_v1` | `cc605fec5c5f` | `mvdec2025_post_skip_latent_bottleneck_v2` |
| A1 | `mvdec_2025_view2_encoder_bottleneck_v1` | `56a24affdb9e` | `mvdec2025_encoder_bottleneck_skip_decoder_v1` |

A0 đặt latent 10D sau toàn bộ dense skip decoder. A1 đặt latent 10D ngay sau
encoder bottleneck 1024D rồi giải mã qua cùng skip decoder. Do hai view được
joint-refine thông qua fused embedding, thay View2 có thể làm trajectory và
final metrics của View1 thay đổi; đó là downstream effect của intervention,
không phải dấu hiệu trộn config.

Artifact A1 đã qua `mvdec-audit-runs`: `1 complete, 0 failed`. Archive tải từ
Kaggle có SHA-256:

```text
d9cf003a2c18a240823af1b9d30547d7d677c5bab186dfd1f0cfa855529a014c
```

## Kết quả ablation

| Representation | A0 ACC | A1 ACC | Delta | A0 NMI | A1 NMI | Delta |
|---|---:|---:|---:|---:|---:|---:|
| View1 | 0.7641 | 0.7644 | +0.0003 | 0.5239 | 0.6037 | +0.0798 |
| View2 | 0.4643 | 0.3843 | -0.0800 | 0.2298 | 0.0944 | -0.1354 |
| Fused | 0.7636 | 0.7643 | +0.0007 | 0.5221 | 0.6033 | +0.0812 |

A0 hội tụ sau 170 updates, A1 sau 160 updates. Runtime lần lượt khoảng 294.8 s
và 304.4 s. Khác biệt số update là kết quả của stopping rule chung trên hai
optimization trajectories, không phải thay đổi training budget.

### Đối chiếu với Table 2 của MvDEC 2025

| Representation | Paper ACC | A0 ACC | A1 ACC | Paper NMI | A0 NMI | A1 NMI |
|---|---:|---:|---:|---:|---:|---:|
| Proposed View1 | 0.7642 | 0.7641 | 0.7644 | 0.5999 | 0.5239 | 0.6037 |
| Proposed View2 | 0.6501 | 0.4643 | 0.3843 | 0.5129 | 0.2298 | 0.0944 |
| Proposed fused | 0.7813 | 0.7636 | 0.7643 | 0.6141 | 0.5221 | 0.6033 |

Paper chỉ báo cáo “best performance”, không công bố seed, standard deviation hay
đầy đủ model-selection protocol. Vì vậy bảng này là diagnostic reference, không
phải paired statistical replication.

## Kết luận được hỗ trợ

1. Chuyển latent View2 về encoder bottleneck không giải quyết khoảng cách fused
   ACC: mức tăng chỉ `0.0007`, tương đương 0.07 điểm phần trăm.
2. A1 làm fused NMI tăng 0.0812, nhưng View2 standalone giảm mạnh cả ACC và NMI.
   Fused improvement vì vậy không chứng minh View2 representation tốt hơn.
3. A1 không nên thay thế protocol chính. Giả thuyết “paper đặt latent tại encoder
   bottleneck và đây là nguyên nhân chính của ACC gap” không được dữ liệu ủng hộ.
4. Một seed không đủ kết luận robustness. Tuy nhiên hiệu ứng âm rất lớn trên
   View2 đủ để loại A1 khỏi ứng viên chính trước khi tốn chi phí multi-seed.

## Chẩn đoán paper: View2 và fusion chưa đủ đặc tả

### 1. View semantics tự mâu thuẫn

Prose p.5 và p.7 nói hai autoencoder học từ hai tập thuộc tính khác nhau. Eq. (3)
lại viết cả hai encoder nhận cùng `x_i`, và Fig. 2 cho cả hai input có 13 chiều.
Implementation hiện tại chọn “hai architecture cùng nhìn một X”. Đây là lựa
chọn bám phương trình/hình, nhưng không thể gọi là natural feature-view
replication.

### 2. Không xác định được latent head của View2

Fig. 2 mô tả nhánh U-Net dense theo chuỗi `13 -> ... -> 1024 -> ... -> 64 ->
23`. Không có node 10D rõ ràng trong View2. Ngược lại, Eq. (3)-(5) yêu cầu một
encoder embedding `h_i^(2)` và một decoder reconstruction riêng.

Hai protocol đã chạy đều thêm một latent 10D mà Fig. 2 không vẽ:

- A0: `64 -> latent 10 -> reconstruction 13`;
- A1: `1024 -> latent 10 -> skip decoder -> reconstruction 13`.

Ablation hiện tại chỉ bác bỏ A1; nó chưa kiểm tra interpretation gần Fig. 2 nhất:
`64 -> linear 23`, sau đó split deterministically thành `h2 = output[:10]` và
`x_hat2 = output[10:]`.

### 3. Fusion trong Eq. (4) và Fig. 2 không giống nhau

Eq. (4) định nghĩa fused embedding là trung bình hai encoded representations:

```text
h = (h1 + h2) / 2
```

Code hiện tại bám Eq. (4) bằng cách average hai latent head tại
`src/representation_learning/MVDEC_dense.py:564`. Tuy nhiên Fig. 2 vẽ mỗi view
cho output 23D rồi `Concatenate` thành output 46D. Hình không nói rõ output 46D
là tensor phục vụ joint reconstruction hay representation đưa vào K-means.
Đoạn prose dưới Fig. 2 còn nói average “as described in Eq. (6)”, trong khi
average thực tế là Eq. (4) và Eq. (6) là K-means loss.

Vì phương trình định nghĩa trực tiếp clustering representation, latent-average
là interpretation chính hợp lý hơn. Full-output concatenate chỉ nên là ablation
“diagram-faithful”, không được gọi là literal paper reproduction.

### 4. Public View2 architecture không được công bố đầy đủ

Fig. 2 chỉ cho ví dụ Air Pollution với input 13 và latent View1 10. Paper không
nêu cách thay đổi View2 cho input 2,000 chiều của REUTERS/20NEWS/RCV1, latent
dimension của View2, initialization, exact learning rate, hay cách lấy
standalone View2 representation từ output 23D. Table 2 vì vậy không thể được tái
lập chính xác chỉ từ mô tả bài báo.

### 5. Kết quả mới khoanh vùng lỗi về View2

View1 ACC của cả A0/A1 gần đúng paper, và View1 NMI của A1 cũng gần paper. Trong
khi đó View2 thấp hơn paper rất lớn ở cả hai architecture. Fused ACC gần như bị
View1 chi phối. Bằng chứng hiện tại hướng tới terminal head/representation
extraction của View2, thay vì dataset, K hoặc vị trí encoder bottleneck.

## Thí nghiệm kế tiếp được đề xuất

### P0: Fig. 2 direct-head ablation - IMPLEMENTED, GPU RUN PENDING

Protocol bất biến đã được implement với tên:

```text
mvdec_2025_view2_direct_23_split_v1
```

Giữ nguyên toàn bộ A0, chỉ thay terminal head của View2:

```text
final skip-decoder feature 64D
        -> Dense(10 + input_dim, linear)
        -> first 10 dimensions: h2
        -> last input_dim dimensions: reconstruction
```

Fusion chính vẫn là Eq. (4): `(h1 + h2) / 2`. Chạy seed 42 trước như diagnostic
gate. Chỉ mở rộng seeds 43/44 nếu View2 cải thiện rõ mà fused metrics không suy
giảm.

Tên protocol giữ `direct_23_split` theo output `10 + 13 = 23` trong Fig. 2. Với
REUTERS, cùng contract tổng quát tạo joint head `10 + 2,000 = 2,010`; exact
input/output dimensions và architecture ID được lưu trong config/manifest.

### P1: fusion sensitivity sau khi khóa View2 head

Nếu P0 vẫn không giải thích gap, so sánh trên cùng final View2 head và seed:

1. `latent_average_10d`: Eq. (4), interpretation chính;
2. `latent_concatenate_20d`: sensitivity của complementary embeddings;
3. `full_output_concatenate_46d`: Fig. 2 diagram diagnostic, không claim Eq. (4).

Mỗi arm phải có protocol/config hash riêng. Không chọn arm tốt nhất trên cùng
test labels rồi báo như confirmatory result; ground truth chỉ dùng ở final
evaluation, và lựa chọn architecture phải được preregister trước multi-seed.

## Trạng thái quyết định

- `mvdec_2025_public_reproduction_v1`: giữ làm reproduction baseline hiện tại.
- `mvdec_2025_view2_encoder_bottleneck_v1`: giữ artifact làm negative/diagnostic
  ablation, không dùng làm main method.
- `mvdec_2025_view2_direct_23_split_v1`: đã implement và khóa contract; đang chờ
  diagnostic run REUTERS seed 42.
