# Intuitive-K-prototypes — Findings cần xử lý

Ngày review: 2026-07-23.
Phạm vi: [src/intuitive_kprototypes/model.py](src/intuitive_kprototypes/model.py) +
lớp bọc [src/pipeline/clustering.py](src/pipeline/clustering.py) +
[configs/intuitive_protocols.json](configs/intuitive_protocols.json), đối chiếu
`docs/2025_Intuitive-K-prototypes- A mixed data clustering algorithm with intuitionistic distribution centroid.pdf`.

**Ngoài phạm vi (thay đổi có chủ đích, không review):** `select_farthest_first_prototypes`,
`empty_cluster_policy="farthest"`, `_repair_small_clusters`, và việc bỏ chuẩn hoá
`numeric_preprocessing: "none"`.

Chưa sửa code nào — đây là danh sách để xử lý dần.

---

## Nền: phần đã kiểm chứng là ĐÚNG

Tái tạo bằng số, không chỉ đọc code:

- **Categorical (Table 2/6/7/8/9)** — chế độ `non_membership="table"` khớp tuyệt đối:
  `E^c` = 0.1429/0.0501/0.4444 (paper 0.143/0.050/0.444); `S^c` = 0.500/0.5117/0.800
  (paper 0.500/0.512/0.800); `φ^c` = 0.3214/0.2809/0.6222 (paper 0.3215/0.281/0.622);
  Eq. (26) distances = .125/.095/.160 và .875/.635/.360 (khớp Table 9 tuyệt đối).
- **Numeric (Table 3/4/5 trên Iris)** — Eq. (5) ξ sepal length .494/.312/.275 khớp;
  ξ* chuẩn hoá 1.000/.632/.557 khớp; Eq. (7)–(12) pair similarity sepal length
  .125/.056/.368 và petal length .011/.005/.087 khớp tới 3 chữ số. Chuỗi bất đối xứng
  `r_global/r_local/d_global/d_local` đúng chiều index.
- Eq. (1), (3), (13)–(14), (16)/(22), (23), (24)–(25), (29), Algorithm 2: đối chiếu tay, khớp.

**Hai chỗ code lệch paper mà lỗi nằm ở paper (không sửa gì):**

1. Table 8 in trọng số 0.4230 — sai làm tròn, ba trọng số cộng lại chỉ được 0.9930.
   Giá trị đúng là 0.4301 (đúng bằng code). Dùng số của paper thì ra lại tổng
   0.354696 của Example 5.
2. Example 3 viết *"similarity of sepal length is 0.283"* — nhưng chính Table 5 của
   paper với θ=1/3 cho (0.125+0.056+0.368)/3 = **0.183**, đúng bằng code.
   → Đừng dùng 0.283 làm mốc test.

---

## 🔴 F1 — Eq. (4) mâu thuẫn với chính Table 2/6/7/8/9 của paper

**Vị trí:** [model.py:447-453](src/intuitive_kprototypes/model.py#L447-L453),
[intuitive_protocols.json:10](configs/intuitive_protocols.json#L10)

Eq. (4) in trong paper dùng mẫu số `|[a^k]|` (tổng số lần xuất hiện của giá trị):

```
υ^k_l = Σ_{s≠l} ( |[a^k] ∩ C_s| / |[a^k]| )²
```

Nhưng Table 2 không dùng công thức đó. `υ_e(C1)` in là **1/100**; với `|[e]|=8`,
`|[e]∩C2|=1` thì Eq. (4) cho `(1/8)² = 1/64`. Chỉ khi mẫu số là `|C2|=10` mới ra
`(1/10)² = 1/100`. Tương tự `υ_d(C2)` in là 9/100 = `(3/10)²` chứ không phải
9/25 = `(3/5)²`. Table 6/7/8/9 đều kế thừa Table 2.

Code có cả hai nhánh; chênh lệch là thật:

```
paper:  E^c = [0.1429, 0.0743, 0.4444]   S^c = [0.500, 0.4417, 0.800]
table:  E^c = [0.1429, 0.0501, 0.4444]   S^c = [0.500, 0.5117, 0.800]  ← khớp paper
```

**Protocol đang chạy `"paper"` → không tái tạo được ví dụ số của paper.**

**Nhưng lựa chọn đó là bắt buộc.** Chạy estimator trên dữ liệu hỗn hợp có cột nhị phân
(đúng dạng FP-Max của project):

```
paper:  ARI=0.9844, hội tụ sau 3 vòng
table:  RuntimeError: IDC violates mu + nu <= 1.
```

Chế độ `table` phá vỡ Property 1 (`0 ≤ μ+υ ≤ 1`) trên dữ liệu nhị phân — mà Property 1
là thứ paper *chứng minh* ở trang 4, và phép chứng minh đó chỉ đúng với mẫu số
`|[a^k]|` của Eq. (4). Kết luận: **Eq. (4) đúng lý thuyết, Table 2 là paper tính sai,
code đang chọn đúng.**

Lưu ý thêm: nhánh `table` còn có quy tắc thứ ba không xuất hiện trong bất kỳ công thức
nào của paper — `inter == 0 → nu = 1` ([model.py:449-451](src/intuitive_kprototypes/model.py#L449-L451)).
Nó cần thiết để khớp `⟨f, 0, 1⟩` trong Table 2, nhưng là suy diễn ngược từ bảng.

### Việc cần làm
- [ ] Giữ nguyên `"non_membership": "paper"`.
- [ ] Thêm 3–4 dòng vào docstring [`compute_intuitionistic_centroids`](src/intuitive_kprototypes/model.py#L396-L402)
      (hiện chỉ ghi *"Compute Definition 1, Eq. (2)-(4)"*): Eq. (4) mâu thuẫn Table 2;
      chọn Eq. (4) vì Property 1; `table` chỉ để tái tạo bảng, không dùng cho dữ liệu nhị phân.
- [ ] Thêm test khoá số Table 8/9 ở chế độ `table` (0.3758/0.4301/0.1941 và
      .125/.095/.160) — đây là bằng chứng mạnh nhất rằng phần categorical cài đúng.

**Vì sao ưu tiên:** không sửa thì người đọc sau tự chạy ví dụ paper, thấy lệch, và
tưởng code sai.

---

## 🟠 F2 — Eq. (5) dùng `|mean|`, cộng với `h_fused` không chuẩn hoá

**Vị trí:** [model.py:523-528](src/intuitive_kprototypes/model.py#L523-L528)

Paper Eq. (5): `ξ = σ/u` nếu `u ≠ 0`. Code dùng `σ/|u|` với ngưỡng `|u| > 1e-12`.

Với dữ liệu paper (chuẩn hoá `[0,1]`, §5.1) `u ≥ 0` nên hai cách giống hệt nhau.
Nhưng protocol đặt `numeric_preprocessing: "none"` trên `h_fused` — latent MvDEC có
giá trị âm và **mean có thể rất gần 0**. Khi đó `ξ` bùng nổ, và vì Eq. (5)–(6) chuẩn
hoá theo `ξ_max` *trên từng thuộc tính*, một cụm có mean ≈ 0 chiếm trọn `ξ_max` và ép
`ξ*` của các cụm khác về ~0. Dây chuyền: `φ^n` méo → Eq. (16) là hàm `φ^{-1}` nên rất
nhạy → trọng số thuộc tính số méo theo.

Không phải lỗi cài đặt công thức, mà là hệ quả của việc bỏ chuẩn hoá — nêu ra vì nó
tương tác trực tiếp với Eq. (5).

### Việc cần làm
- [ ] Không đổi công thức.
- [ ] Ghi `numeric_complexity` cực đại (hoặc tỉ lệ `|mean| < ngưỡng`) vào diagnostics,
      cạnh `final_numeric_phi` đã có sẵn ở [clustering.py:452-456](src/pipeline/clustering.py#L452-L456).
      Nếu `ξ` nhảy vài bậc độ lớn thì biết ngay trọng số đang bị một cụm chi phối.
      Chi phí gần bằng 0, chỗ ghi đã có sẵn.

---

## 🟠 F3 — Thứ tự bước 7/8 của Algorithm 2: code chọn đúng, thiếu chú thích

**Vị trí:** [`_state_from_labels`](src/intuitive_kprototypes/model.py#L1042-L1060)

Algorithm 2 (trang 8) ghi dòng 7 *cập nhật trọng số* rồi dòng 8 *cập nhật IDC*.
Nhưng §4.4 Step 3 ghi ngược: cập nhật IDC **rồi** cập nhật trọng số.

Bản trong Algorithm 2 không chạy được ở vòng đầu, vì Eq. (17)–(22) cần `q_{l j_c}` mà
IDC lúc đó chưa tồn tại. Code tính tâm trước, trọng số sau — theo §4.4, là cách đọc
duy nhất chạy được.

### Việc cần làm
- [ ] Thêm một dòng comment ghi rõ đã giải quyết mâu thuẫn Algorithm 2 dòng 7-8 vs
      §4.4 Step 3 theo hướng nào và tại sao.

---

## 🟡 F4 — `mu_param` mặc định 0.5 nằm ngoài dải paper khuyến nghị

**Vị trí:** [model.py:883](src/intuitive_kprototypes/model.py#L883),
[clustering.py:358](src/pipeline/clustering.py#L358)

§5.2 kết luận rõ: *"the parameter value should be between 0 and 0.4"*, kèm Fig. 5 cho
thấy metric rơi vực khi `μ` vượt ~0.4. Table 13 (BCW) dùng `μ = 0.2`.

Grid quét 0.1–0.9 ([intuitive_protocols.json:45](configs/intuitive_protocols.json#L45))
thì không sai — có silhouette làm trọng tài, quét rộng hơn khuyến nghị là hợp lệ.
Vấn đề chỉ ở **giá trị mặc định**, thứ mà mọi lời gọi không truyền tham số sẽ nhận.

### Việc cần làm
- [ ] Đổi mặc định `mu_param` 0.5 → 0.2 ở cả hai chỗ.

---

## 🟡 F5 — Cột nhị phân hằng số vẫn tác động lên phép gán cụm

**Vị trí:** [clustering.py:381-382](src/pipeline/clustering.py#L381-L382),
[intuitive_protocols.json:25](configs/intuitive_protocols.json#L25)

Grid FP-Max có `min_support: 1.0` → sinh được itemset xuất hiện ở **mọi** dòng → cột
toàn `"1"`, `t = 1`.

Với cột như vậy Eq. (24) cho `d1 = Σ_k μ^k − μ^x = 0`, nhưng `d2 = ν^x ≠ 0`
**và khác nhau giữa các cụm**. Đo thực tế:

```
cột hằng số:  mu = [0.302, 0.320, 0.378]   nu = [0.245, 0.234, 0.194]
```

Khoảng cách `(1−μ_param)·ν` chênh 0.245 vs 0.194 giữa cụm 1 và cụm 3 → một thuộc tính
**không mang thông tin nào** vẫn kéo mọi điểm về cụm lớn nhất, lại còn được cấp trọng
số khác 0 qua Eq. (21)–(22).

Đây là đặc tính của chính công thức paper. Paper không gặp vì §5.1 nói rõ
*"attributes with only one value are removed"*. Code đang không lọc.

### Việc cần làm
- [ ] Lọc cột nhị phân hằng số trước khi đưa vào `IntuitiveKPrototypes`
      (chỗ tự nhiên: [clustering.py:381-382](src/pipeline/clustering.py#L381-L382)).
      Vừa đúng §5.1, vừa loại một nguồn nhiễu có thật.

**Đây là mục duy nhất trong danh sách có ảnh hưởng thật lên nhãn đầu ra.**

---

## 🟡 F6 — Ghi chú diễn giải: Eq. (23) cân bằng hai view theo *số lượng* thuộc tính

`num*` tổng bằng `m_n`, `cat*` tổng bằng `m_c`, hai nhóm không chuẩn hoá chung
([model.py:794-795](src/intuitive_kprototypes/model.py#L794-L795)). Đúng paper.

Hệ quả trong project này: 10 chiều latent + 30 đặc trưng FP-Max → view phân loại tự
động nhận gấp 3 tổng khối lượng trọng số. Và `m_c` **thay đổi theo từng cấu hình
`min_support` trong grid**.

### Việc cần làm
- [ ] Không sửa code. Khi so sánh các cấu hình có số đặc trưng khác nhau, nhớ rằng tỉ lệ
      hai view cũng đang thay đổi theo — cần nói rõ điều này khi diễn giải kết quả grid.

---

## 🟡 F7 — Các điểm nhỏ

- **Algorithm 1 mơ hồ về thứ tự chọn.** Dòng 18 *"select x^i whose distances satisfy…"*
  không nói chọn cái nào khi có nhiều ứng viên; code lấy chỉ số nhỏ nhất
  ([model.py:315-317](src/intuitive_kprototypes/model.py#L315-L317)) → kết quả phụ thuộc
  thứ tự dòng dữ liệu. Không ảnh hưởng protocol hiện tại (đang dùng `farthest_first`),
  nhưng cần biết nếu sau này bật lại `init_strategy="paper"`.
- **`IterationRecord.cost`** ([model.py:993](src/intuitive_kprototypes/model.py#L993))
  tính `Φ` theo `Q,S` **mới** với nhãn `U` **cũ**. Đó đúng là Eq. (27) tại trạng thái đó,
  nhưng vì `Q,S` đổi mỗi vòng nên dãy này không đảm bảo giảm đơn điệu — **đừng dùng nó
  để chẩn đoán hội tụ**. Hội tụ đang dựa trên `changed == 0`, khớp Algorithm 2 dòng 16
  và khớp `stopping_mode: "exact_assignment"`. Phần này đúng.

---

## 🟡 F8 — Hiệu năng (ngoài phạm vi fidelity)

[`compute_intuitionistic_centroids`](src/intuitive_kprototypes/model.py#L418-L454) và
[`categorical_attribute_distances`](src/intuitive_kprototypes/model.py#L827-L837) chạy
vòng lặp Python lồng 3–4 lớp, gọi lại **mỗi vòng lặp của mỗi lần fit**. Grid hiện tại:
324 bộ tham số × 171 cấu hình FP-Max.

Docstring module nói rõ đây là đánh đổi có chủ đích lấy khả năng truy vết — hợp lý.
Nhưng nếu thời gian chạy thành nút cổ chai thì đó là chỗ đầu tiên nên vector hoá, và
**vector hoá được mà không đụng tới công thức**.

---

## Thứ tự đề xuất xử lý

| # | Mục | Lý do xếp trước |
|---|---|---|
| 1 | **F5** lọc cột nhị phân hằng số | Lỗi thật, ảnh hưởng thật lên nhãn, sửa ở một chỗ |
| 2 | **F1** tài liệu + test khoá số Table 8/9 | Giữ lại phát hiện, tránh người sau hiểu nhầm code sai |
| 3 | **F4** đổi mặc định `mu_param` → 0.2 | Một dòng, đúng khuyến nghị paper |
| 4 | **F2** diagnostic cho ξ cực đại | Chi phí ~0, chỗ ghi đã có sẵn |
| 5 | **F3** comment thứ tự bước 7/8 | Một dòng |
| 6 | **F6 / F7 / F8** | Ghi chú, không chặn gì |

Không mục nào đòi viết lại công thức.
