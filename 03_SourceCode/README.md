# Mã nguồn – Dự đoán giá nhà Ames Housing (BTL Học máy 608807 - 69CS1)

Hệ thống pipeline học máy hoàn chỉnh cho bài toán định giá bất động sản Ames Housing, tuân thủ nguyên tắc **chống rò rỉ dữ liệu (No Data Leakage)** tuyệt đối trong Cross-Validation và hỗ trợ đóng gói mô hình scikit-learn sang `.joblib` để suy luận độc lập.

---

## 1. Cấu trúc thư mục

```text
03_SourceCode/
├── data/
│   ├── raw/
│   │   └── train.csv                       # Dữ liệu gốc Kaggle (1460 dòng x 81 cột)
│   └── processed/
│       └── test_ids.csv                    # 292 Id tập test cố định dùng chung cho nhóm
├── models/
│   └── best_model.joblib                   # Pipeline tối ưu đã huấn luyện (Lasso)
├── train/
│   └── train_demo.py                       # Script so sánh 7 mô hình qua 5-Fold CV & xuất joblib
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb          # Khám phá dữ liệu và tiền xử lý (Tuấn Anh)
│   ├── 02_Modeling_Evaluation.ipynb        # Đánh giá mô hình & 5-Fold CV (Hinh)
│   └── drafts/                             # Bản nháp thử nghiệm đặc trưng
├── src/
│   ├── __init__.py                         # Export module dùng chung
│   ├── data.py                             # Hàm nạp, chia train/test cố định, remove_outliers()
│   ├── transformers.py                     # MissingHandler, OrdinalMapper, FeatureAdder
│   ├── pipeline.py                         # build_pipeline(), build_preprocessor(), get_models()
│   └── plots.py                            # plot_model_evaluation() vẽ biểu đồ 4 góc
├── app/
│   └── app.py                              # Demo Webapp Streamlit
├── predict.py                              # CLI & hàm suy luận cho dữ liệu CSV mới
├── .flake8                                 # Cấu hình kiểm tra chuẩn PEP8
└── requirements.txt                        # Danh sách thư viện cần thiết
```

*Tài liệu kỹ thuật và biên bản họp được lưu trữ tại thư mục `docs/` ở gốc dự án (ví dụ: `docs/Demo.md`, `docs/Bien_ban_hop_25_11.md`).*

---

## 2. Kiến trúc Pipeline 5 bước

Toàn bộ quá trình tiền xử lý và học máy được tích hợp trong một đối tượng `TransformedTargetRegressor(Pipeline)`:

1. **`MissingHandler`**:
   - **Nhóm A (NA = "None" / 0)**: 15 cột phân loại (`PoolQC`, `GarageType`, `BsmtQual`, `BsmtFinType1/2`...) điền `"None"`; 9 cột số tương ứng (`GarageArea`, `TotalBsmtSF`...) điền `0`. `GarageYrBlt` điền bằng `YearBuilt`.
   - **Nhóm B (Khuyết thật)**: `LotFrontage` điền median theo từng khu `Neighborhood` (học từ fold train; khu lạ fallback về median chung). `Electrical`, `BsmtExposure`, `BsmtFinType2` điền mode học từ fold train (riêng tầng hầm chỉ điền mode khi nhà có tầng hầm).
2. **`OrdinalMapper`** (bước `"ordinal"`, dùng `"passthrough"` khi `ordinal=False` để phục vụ tuning):
   - Mã hóa thứ bậc **12 cột** chất lượng (`ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `BsmtExposure`, `GarageFinish`, `CentralAir`).
   - `BsmtFinType1` và `BsmtFinType2` giữ nguyên dạng chuỗi để đi vào nhánh One-Hot Encoding.
3. **`FeatureAdder` (Feature Engineering)**:
   - Tạo các cột tổng hợp: `TotalSF`, `HouseAge` (kẹp $\ge 0$), `RemodAge`, `TotalBath`, `TotalPorchSF`.
   - Cờ nhị phân: `HasPool`, `Has2ndFloor`, `HasGarage`, `HasFireplace`.
   - Ép kiểu phân loại cho cột số nhãn: `MSSubClass`, `MoSold`, `YrSold`.
   - Bỏ cột `PoolQC` sau khi tạo `HasPool` (vì 1.453/1.460 căn không có hồ bơi).
4. **`ColumnTransformer`** (3 nhánh theo thứ tự `num`, `cat`, `log`):
   - **Nhánh `num`** (các cột số trừ `SKEWED_COLS`): `SimpleImputer(median)` $\to$ `StandardScaler()`.
   - **Nhánh `cat`** (các cột phân loại, bao gồm `BsmtFinType1/2`): `SimpleImputer("None")` $\to$ `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)`.
   - **Nhánh `log`** (8 biến lệch phải `SKEWED_COLS`: `LotArea`, `GrLivArea`, `1stFlrSF`, `LotFrontage`, `OpenPorchSF`, `MasVnrArea`, `WoodDeckSF`, `2ndFlrSF`): `SimpleImputer(median)` $\to$ `FunctionTransformer(np.log1p, feature_names_out="one-to-one")` $\to$ `StandardScaler()`.
5. **Mô hình hồi quy** (`Dummy`, `Ridge`, `Lasso`, `KNN`, `DecisionTree`, `RandomForest`, `HistGradientBoosting`) được bọc trong `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` để tối ưu chuẩn độ đo RMSLE và tự động chuyển về đơn vị USD khi dự đoán.

---

## 3. Cài đặt môi trường

```bash
# Khởi tạo môi trường ảo Python
python3 -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 4. Hướng dẫn sử dụng

### 4.1. Huấn luyện và so sánh mô hình (`train/train_demo.py`)

Chạy đánh giá so sánh 7 mô hình bằng **5-Fold Cross Validation** trên tập train và lưu mô hình tối ưu vào `models/best_model.joblib`:

```bash
# Chạy với mã hóa thứ bậc Ordinal (mặc định: Lasso RMSLE 0.1086, Ridge 0.1101)
python train/train_demo.py

# Chạy với "passthrough" ở bước ordinal để one-hot toàn bộ cột chất lượng
python train/train_demo.py --no-ordinal
```

### 4.2. Khám phá và đánh giá trên Notebook (`notebooks/02_Modeling_Evaluation.ipynb`)

Mở notebook `notebooks/02_Modeling_Evaluation.ipynb` để theo dõi chi tiết:
- Sơ đồ Pipeline 5 bước (kèm 3 nhánh `num`, `cat`, `log`).
- Bảng kết quả 5-Fold Cross Validation định lượng trên tập train sạch (1.165 dòng).
- Bảng đối chứng giữa `OrdinalMapper` và `"passthrough"` (One-Hot Encoding).
- Nhận xét chi tiết phục vụ viết báo cáo kỹ thuật.

### 4.3. Suy luận dữ liệu mới (`predict.py`)

Chạy dự đoán giá nhà từ file CSV thô (kết quả lưu vào file CSV chỉ định):

```bash
# Dự đoán cho tập dữ liệu bất kỳ
python predict.py --input data/raw/train.csv --output predictions.csv
```

Sử dụng trong mã Python / Notebook / Streamlit:

```python
import pandas as pd
from predict import predict

df = pd.read_csv("data/raw/train.csv", keep_default_na=False, na_values=["NA", ""])
predictions = predict(df)  # Trả về pd.Series giá nhà USD
print(predictions.head())
```
