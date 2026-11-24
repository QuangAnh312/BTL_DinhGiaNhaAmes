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
│   ├── 02_Modeling_Evaluation.ipynb         # Đánh giá mô hình & 5-Fold CV (Hinh)
│   └── drafts/                             # Bản nháp thử nghiệm đặc trưng
├── src/
│   ├── __init__.py                         # Export module dùng chung
│   ├── data.py                             # Hàm nạp và phân chia train/test cố định, lọc ngoại lai
│   ├── transformers.py                     # MissingHandler, OrdinalMapper, FeatureAdder
│   ├── pipeline.py                         # build_pipeline(), build_preprocessor(), get_models()
│   └── plots.py                            # plot_model_evaluation() vẽ biểu đồ 4 góc
├── app/
│   └── app.py                              # Demo Webapp Streamlit
├── predict.py                              # CLI & hàm suy luận cho dữ liệu CSV mới
└── requirements.txt                        # Danh sách thư viện cần thiết
```

*Tài liệu kỹ thuật và báo cáo được lưu trữ tại thư mục `docs/` ở gốc dự án (ví dụ: `docs/Demo.md`, `docs/Huong_xu_ly_du_lieu_va_cap_nhat_Pipeline.docx`).*

---

## 2. Kiến trúc Pipeline 5 bước

Toàn bộ quá trình tiền xử lý và học máy được tích hợp trong một đối tượng `TransformedTargetRegressor(Pipeline)`:

1. **`MissingHandler`**:
   - **Nhóm A (NA = "None" / 0)**: 15 cột phân loại (`PoolQC`, `GarageType`, `BsmtQual`...) điền `"None"`; 9 cột số tương ứng (`GarageArea`, `TotalBsmtSF`...) điền `0`. `GarageYrBlt` điền bằng `YearBuilt`.
   - **Nhóm B (Khuyết thật)**: `LotFrontage` điền median theo từng khu `Neighborhood` (học từ fold train; khu lạ fallback về median chung). `Electrical`, `BsmtExposure`, `BsmtFinType2` điền mode học từ fold train (riêng tầng hầm chỉ điền mode khi nhà có tầng hầm).
2. **`OrdinalMapper` (Tùy chọn)**: Mã hóa thứ bậc 12 cột chất lượng (`ExterQual`, `KitchenQual`, `BsmtQual`, `GarageFinish`...) thành thang điểm nguyên $0 \to 5$.
3. **`FeatureAdder` (Feature Engineering)**:
   - Tạo các cột tổng hợp: `TotalSF`, `HouseAge` (kẹp $\ge 0$), `RemodAge`, `TotalBath`, `TotalPorchSF`.
   - Cờ nhị phân: `HasPool`, `Has2ndFloor`, `HasGarage`, `HasFireplace`.
   - Ép kiểu phân loại cho cột số nhãn: `MSSubClass`, `MoSold`, `YrSold`.
4. **`ColumnTransformer`**:
   - Nhánh số: `SimpleImputer(median)` $\to$ `StandardScaler()`.
   - Nhánh phân loại: `SimpleImputer("None")` $\to$ `OneHotEncoder(min_frequency=0.01, sparse_output=False)`.
5. **Mô hình hồi quy** (Dummy, Ridge, Lasso, KNN, DecisionTree, RandomForest, HistGradientBoosting) được bọc trong `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` để tối ưu chuẩn độ đo RMSLE và tự động chuyển về đơn vị USD khi dự đoán.

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
# Chạy với mã hóa thứ bậc Ordinal (mặc định)
python train/train_demo.py

# Chạy với One-Hot Encoding cho toàn bộ cột chất lượng (để so sánh)
python train/train_demo.py --no-ordinal
```

### 4.2. Khám phá và đánh giá trên Notebook (`notebooks/02_Modeling_Evaluation.ipynb`)

Mở notebook `notebooks/02_Modeling_Evaluation.ipynb` để theo dõi chi tiết:
- Sơ đồ Pipeline 5 bước.
- Bảng kết quả 5-Fold Cross Validation định lượng trên tập train sạch (1.165 dòng).
- Bảng so sánh giữa Ordinal Encoding và One-Hot Encoding.
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

df = pd.read_csv("data/raw/train.csv")
predictions = predict(df)  # Trả về pd.Series giá nhà USD
print(predictions.head())
```
