# Biên bản chốt chiến lược tiền xử lý – Buổi 3 (T4 25/11/2026)

**Nội dung:** review EDA (notebook 01) và chốt chiến lược tiền xử lý cho Pipeline (notebook 02).
**Người chuẩn bị:** Ngô Quang Anh (nhóm trưởng).
**Trạng thái:** đã chốt bằng CV trên tập train; cả nhóm xác nhận tại buổi họp. Việc cài vào code do
người phụ trách từng phần thực hiện (Mục 6).

Tài liệu nội bộ, không nộp. Mọi con số dưới đây lấy từ output thật của các script trong
`docs/thi_nghiem_25_11/` và notebook 02; cách chạy lại ở Mục 5.

---

## 1. Review EDA (notebook 01, Tuấn Anh)

**Đạt yêu cầu Checkpoint 1.** Notebook chạy *Restart & Run All* không lỗi trên Python 3.11; các con số
trong nhận xét khớp output (đã kiểm tra độc lập: tương quan, η², Spearman, trung vị theo bậc).

- Không rò rỉ dữ liệu: mọi thống kê chỉ tính trên tập train; mục 6–8 dùng train đã bỏ ngoại lai
  (1.165 căn); không `fit` mô hình hay transformer nào.
- Điểm mạnh: kiểm chứng "NA = không có" bằng cột diện tích/số lượng (Mục 4.2); giải thích vì sao
  `BsmtFinType1 = Unf` lại đắt (lẫn với tuổi và chất lượng nhà, Mục 8.2); chỉ ra η² bị thổi phồng ở
  biến nhiều nhóm nhỏ (Mục 7.1).
- 6 điểm còn bỏ ngỏ ở Mục 8.5, cộng các điểm lệch giữa EDA và Pipeline của Hinh, được chốt ở Mục 3.

## 2. Cách chốt

- Chỉ dùng **tập train** (1.165 căn sau khi bỏ ngoại lai). **Không dùng tập test.**
- `KFold(n_splits=5, shuffle=True, random_state=42)`, cùng các fold cho mọi phương án; chỉ số RMSLE.
- Mỗi phương án đổi **đúng một điểm** so với Pipeline ban đầu (commit `6d32e2e`), chạy với 3 mô hình:
  Ridge (`alpha=10`), Lasso (`alpha=0.001`), HistGB (mặc định).
- Quy tắc: nhận thay đổi khi mô hình tuyến tính (Ridge, Lasso – hai mô hình tốt nhất) tốt hơn đều
  qua các fold và HistGB không kém đi; kết quả trái chiều hoặc không đổi thì giữ cách đơn giản hơn.
- Độ lệch chuẩn RMSLE giữa các fold của Pipeline ban đầu: Ridge 0,0134 · Lasso 0,0156 · HistGB 0,0157.

## 3. Kết quả từng điểm

Chênh RMSLE so với Pipeline ban đầu (âm là tốt hơn); trong ngoặc là số fold tốt hơn trên 5.
Pipeline ban đầu: Ridge 0,1138 · Lasso 0,1106 · HistGB 0,1231.

| # | Phương án | Ridge | Lasso | HistGB | Chốt |
|---|---|---|---|---|---|
| 1 | `BsmtFinType1/2` → one-hot | −0,0001 (3) | −0,0001 (4) | −0,0008 (4) | **Nhận** – khớp EDA Mục 8.2 |
| 2 | `log1p` cho 8 biến lệch | −0,0033 (5) | −0,0019 (5) | 0,0000 (2) | **Nhận** |
| 3 | Bỏ `Utilities`, `Street` | 0,0000 (2) | 0,0000 (2) | 0,0000 (0) | Không – không đổi gì; `min_frequency` đã gộp mức hiếm |
| 4 | Bỏ `PoolQC`, giữ `HasPool` | −0,0005 (4) | −0,0002 (3) | 0,0000 (0) | **Nhận** |
| 5 | `MoSold`, `YrSold` dạng số | −0,0008 (4) | +0,0001 (1) | −0,0006 (3) | Không – trái chiều (xem Mục 3.2) |
| 6 | Bỏ `GarageYrBlt` | −0,0001 (4) | 0,0000 (3) | +0,0009 (1) | Không – giữ điền `YearBuilt` |
| 7 | `min_frequency` = 10 căn | 0,0000 (0) | 0,0000 (0) | 0,0000 (0) | Không – y hệt 0,01 |
| 8 | Không gộp mức hiếm | −0,0004 (4) | +0,0001 (2) | −0,0002 (1) | Không – trái chiều |
| 9 | Thêm `HasBsmt`, `HasRemodel` | +0,0002 (2) | +0,0001 (3) | −0,0006 (4) | Không – không giúp Ridge/Lasso |

Bảng đầy đủ (giá trị chưa làm tròn): `docs/thi_nghiem_25_11/ket_qua_so_sanh_phuong_an.csv`.

### 3.1. Ngưỡng ngoại lai

Không so được bằng CV thường (bỏ dòng thì tập đánh giá cũng đổi). Cách so công bằng: bỏ ngoại lai
**chỉ ở phần huấn luyện của mỗi fold**, phần đánh giá giữ đủ 1.168 căn train.

| Quy tắc | Ridge | Lasso | HistGB |
|---|---|---|---|
| Không bỏ | 0,1499 | 0,1499 | 0,1302 |
| **`GrLivArea > 4000` (3 căn)** | **0,1441** | 0,1429 | **0,1292** |
| `> 4000` và giá < 300.000 (2 căn) | 0,1446 | 0,1428 | 0,1313 |

→ **Giữ `GrLivArea > 4000`**: tốt hơn rõ so với không bỏ; ngang quy tắc 2 căn với Lasso, tốt hơn với
Ridge và HistGB; đúng khuyến nghị của De Cock (2011), dễ giải thích.

### 3.2. Gộp các thay đổi được nhận (1 + 2 + 4)

| Cấu hình | Ridge | Lasso | HistGB |
|---|---|---|---|
| Pipeline ban đầu | 0,1138 ± 0,0134 | 0,1106 ± 0,0156 | 0,1231 ± 0,0157 |
| **Đã chốt: log1p + BsmtFin one-hot + bỏ PoolQC** | **0,1101 ± 0,0133** | **0,1086 ± 0,0157** | **0,1223 ± 0,0140** |
| Đã chốt + `MoSold`/`YrSold` dạng số | 0,1093 ± 0,0132 | 0,1085 ± 0,0156 | 0,1243 ± 0,0142 |

Thêm `MoSold`/`YrSold` dạng số làm HistGB kém đi 0,0020 → giữ dạng phân loại.

### 3.3. Ordinal hay One-Hot cho 12 cột thứ bậc

Đo thử trên Pipeline đã chốt (bản cài thử để kiểm chứng, chưa đưa vào repo), tắt `OrdinalMapper`
(one-hot): Lasso 0,1086 → 0,1073, Ridge 0,1101 → 0,1085
(Ridge tốt hơn 5/5 fold, Lasso 4/5), nhưng HistGB 0,1223 → 0,1230 (tốt hơn 3/5 fold).
→ Trái chiều giữa các mô hình và phụ thuộc `alpha` chưa tinh chỉnh: **giữ Ordinal làm mặc định, đưa
vào tuning** dưới dạng siêu tham số `regressor__ordinal` ∈ {`OrdinalMapper()`, `"passthrough"`}.

## 4. Chiến lược tiền xử lý đã chốt (Hinh cài vào `src/`)

So với Pipeline hiện tại trên `main`, cần đổi 3 chỗ: thêm nhánh `log1p` cho 8 biến lệch trong
`build_preprocessor()`; bỏ `BsmtFinType1`, `BsmtFinType2`, `PoolQC` khỏi `ORDINAL_MAPS`; bỏ cột `PoolQC`
sau `FeatureAdder`. Các bước còn lại giữ nguyên.

| Bước | Nội dung | Học từ dữ liệu? |
|---|---|---|
| 0 | Bỏ 3 căn `GrLivArea > 4000` **chỉ trên train**, ngoài Pipeline (`remove_outliers`) | Không |
| 1 | `MissingHandler`: NA "không có" → `"None"`/0; `GarageYrBlt` ← `YearBuilt`; `LotFrontage` ← trung vị theo `Neighborhood`; `Electrical` ← mode | **Có** (trung vị, mode) |
| 2 | `OrdinalMapper`: 9 cột thang Po–Ex, `BsmtExposure`, `GarageFinish`, `CentralAir` (`BsmtFinType1/2` → one-hot) | Không |
| 3 | `FeatureAdder`: `TotalSF`, `HouseAge`, `RemodAge`, `TotalBath`, `TotalPorchSF`, `HasPool`, `Has2ndFloor`, `HasGarage`, `HasFireplace`; `MSSubClass`/`MoSold`/`YrSold` → chuỗi; bỏ `PoolQC` | Không |
| 4a | Nhánh số: median → `StandardScaler` | **Có** |
| 4b | Nhánh 8 biến lệch: median → `log1p` → `StandardScaler` | **Có** |
| 4c | Nhánh phân loại: `"None"` → `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)` | **Có** |
| 5 | Mô hình trong `TransformedTargetRegressor(log1p / expm1)` | **Có** |

Nguồn công thức đặc trưng duy nhất là `FeatureAdder` trong `src/transformers.py`
(`src/feature_proposal.py` của Hoàng Anh chỉ để minh họa, không dùng trong mô hình).
Khác biệt nhỏ: `RemodAge` âm (1 căn ghi năm cải tạo sau năm bán) được kẹp về 0.

## 5. Cách chạy lại số liệu

- Bảng Mục 3 và 3.1–3.2: chạy 2 script trong `docs/thi_nghiem_25_11/` trên commit `6d32e2e`
  (Pipeline trước khi chốt) – hướng dẫn ở đầu file `so_sanh_phuong_an.py`. Đã chạy lại, kết quả khớp.
- **Tiêu chí nghiệm thu khi Hinh cài xong:** notebook 02 (Restart & Run All) hoặc
  `python train/train_demo.py` phải cho đúng Ridge 0,1101 · Lasso 0,1086 · HistGB 0,1223 như dòng
  "Đã chốt" ở Mục 3.2 (cùng fold, cùng siêu tham số).

## 6. Việc tiếp theo

| Việc | Người | Hạn |
|---|---|---|
| Cài chiến lược Mục 4 vào `src/`; chạy lại notebook 02 + `train/train_demo.py` (`best_model.joblib`), sửa nhận xét theo số mới; cập nhật `docs/Demo.md` và `03_SourceCode/README.md` (Demo.md còn mô tả file đã xóa) | Hinh | T5 26/11 |
| Notebook 01: (1) thêm 1 ô markdown "Kết quả chốt" ngay sau Mục 8.5 (tóm tắt bảng Mục 3 biên bản), để các câu hỏi ở 8.5 có kết luận; (2) bỏ `scipy` – vẽ Q-Q plot bằng `statistics.NormalDist().inv_cdf` (thư viện chuẩn Python) thay cho `stats.probplot`, vì nhóm chỉ dùng sklearn/pandas/numpy/matplotlib/seaborn và `scipy` không có trong `requirements.txt` | Tuấn Anh | T5 26/11 |
| Báo cáo tiến độ 01: sơ đồ EDA (`01_Report/figures/`) + bảng chiến lược ở Mục 4 | Tiến Quang (tổng hợp từ Tuấn Anh, Hinh) | T5 26/11 |
| Review leakage, notebook *Restart & Run All*, merge `main` | Quang Anh | T6 27/11 |
| Tuning (`RandomizedSearchCV`, gồm `alpha` và `regressor__ordinal`), Learning/Validation Curve | Quang Anh | CN 13/12 |
| Phân tích lỗi, feature importance trên Pipeline đã chốt | Hoàng Anh | theo README |
| Checkpoint 1 – GV kiểm tra | Cả nhóm | T4 02/12 |

Mỗi thành viên đọc Mục 3–4 để trả lời được khi vấn đáp: *vì sao lấy log1p cho 8 biến, vì sao
`BsmtFinType` không mã hóa thứ bậc, vì sao bỏ ngoại lai ngoài Pipeline, vì sao không dùng tập test để chọn*.
