# Mã nguồn – Dự đoán giá nhà Ames Housing (BTL Học máy 608807 - 69CS1)

Hệ thống pipeline học máy hoàn chỉnh cho bài toán định giá bất động sản Ames Housing, tuân thủ nguyên tắc **chống rò rỉ dữ liệu (No Data Leakage)** tuyệt đối trong Cross-Validation và hỗ trợ đóng gói mô hình scikit-learn sang `.joblib` để suy luận độc lập.

---

## 1. Cấu trúc thư mục chuẩn hóa

```text
03_SourceCode/
├── data/
│   ├── raw/
│   │   └── train.csv                       # Dữ liệu gốc Kaggle (1460 dòng x 81 cột)
│   └── processed/
│       └── test_ids.csv                    # 292 Id tập test cố định dùng chung cho nhóm
├── docs/
│   └── Huong_xu_ly_du_lieu_va_cap_nhat_Pipeline.docx  # Thiết kế kỹ thuật chi tiết
├── models/
│   └── best_model.joblib                   # Pipeline tối ưu đã huấn luyện (Lasso)
├── train/
│   ├── train_demo.py                       # Script so sánh 7 mô hình qua 5-Fold CV & xuất joblib
│   └── train_all_models.py                 # Huấn luyện toàn bộ 7 mô hình, xuất CSV và sơ đồ đánh giá
├── output/
│   ├── GiaNha_Dummy.csv                    # Dự đoán và phân tích sai số mô hình Dummy (Baseline)
│   ├── GiaNha_Ridge.csv                    # Dự đoán và phân tích sai số mô hình Ridge
│   ├── GiaNha_Lasso.csv                    # Dự đoán và phân tích sai số mô hình Lasso (Tối ưu)
│   ├── GiaNha_KNN.csv                      # Dự đoán và phân tích sai số mô hình KNN
│   ├── GiaNha_DecisionTree.csv             # Dự đoán và phân tích sai số mô hình Decision Tree
│   ├── GiaNha_RandomForest.csv             # Dự đoán và phân tích sai số mô hình Random Forest
│   ├── GiaNha_HistGB.csv                   # Dự đoán và phân tích sai số mô hình HistGradientBoosting
│   ├── GiaNha_DuDoan.csv                   # Kết quả suy luận dữ liệu mới từ predict.py
│   └── plots/                              # Sơ đồ đánh giá 4 góc trực quan (.png)
│       ├── DanhGia_Dummy.png
│       ├── DanhGia_Ridge.png
│       ├── DanhGia_Lasso.png
│       ├── DanhGia_KNN.png
│       ├── DanhGia_DecisionTree.png
│       ├── DanhGia_RandomForest.png
│       └── DanhGia_HistGB.png
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb          # Khám phá dữ liệu và tiền xử lý
│   ├── 02_Modeling_Evaluation.ipynb         # Đánh giá mô hình & K-Fold CV
│   └── train_models/                       # Folder chứa 7 notebook chạy train thử từng mô hình
│       ├── 01_Train_Dummy.ipynb
│       ├── 02_Train_Ridge.ipynb
│       ├── 03_Train_Lasso.ipynb
│       ├── 04_Train_KNN.ipynb
│       ├── 05_Train_DecisionTree.ipynb
│       ├── 06_Train_RandomForest.ipynb
│       └── 07_Train_HistGB.ipynb
├── src/
│   ├── __init__.py                         # Export module dùng chung
│   ├── data.py                             # Hàm nạp và phân chia train/test cố định
│   ├── transformers.py                     # MissingHandler, OrdinalMapper, FeatureAdder
│   └── pipeline.py                         # build_pipeline(), build_preprocessor(), get_models()
├── app/
│   └── app.py                              # Demo Webapp Streamlit
├── predict.py                              # CLI & hàm suy luận cho dữ liệu CSV mới
└── requirements.txt                        # Danh sách thư viện cần thiết
```

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

Chạy đánh giá so sánh 7 mô hình bằng **5-Fold Cross Validation** và lưu mô hình tối ưu vào `models/best_model.joblib`:

```bash
# Chạy với mã hóa thứ bậc Ordinal (mặc định)
python train/train_demo.py

# Chạy với One-Hot Encoding cho toàn bộ cột chất lượng (để so sánh)
python train/train_demo.py --no-ordinal

# Lọc ngoại lai theo điều kiện Dean De Cock
python train/train_demo.py --outlier-mode de_cock
```

### 4.2. Huấn luyện toàn bộ 7 mô hình và xuất đồ thị (`train/train_all_models.py`)

Chạy lệnh sau để huấn luyện độc lập cả 7 mô hình, xuất ra 7 file kết quả dự đoán `output/GiaNha_<TênMôHình>.csv` và 7 sơ đồ trực quan `output/plots/DanhGia_<TênMôHình>.png`:

```bash
python train/train_all_models.py
```

### 4.3. Chạy từng mô hình riêng lẻ bằng Notebook (`notebooks/train_models/`)

Mở các file `.ipynb` trong thư mục `notebooks/train_models/` (hoặc `train_notebooks/`):
* `01_Train_Dummy.ipynb`: Mô hình cơ sở Baseline.
* `02_Train_Ridge.ipynb`: Hồi quy Ridge với điều chuẩn L2.
* `03_Train_Lasso.ipynb`: Hồi quy Lasso với điều chuẩn L1 (Mô hình tốt nhất).
* `04_Train_KNN.ipynb`: Thuật toán K láng giềng gần nhất.
* `05_Train_DecisionTree.ipynb`: Cây quyết định CART.
* `06_Train_RandomForest.ipynb`: Rừng ngẫu nhiên (300 cây đóng bao).
* `07_Train_HistGB.ipynb`: Histogram-based Gradient Boosting.

Mỗi notebook đều có đủ code nạp dữ liệu, kiểm định 5-fold CV, đánh giá trên test set, vẽ sơ đồ và xuất file CSV tương ứng vào `output/`.

### 4.4. Suy luận dữ liệu mới (`predict.py`)

Chạy dự đoán giá nhà từ file CSV thô (kết quả mặc định lưu vào `output/GiaNha_DuDoan.csv` với đầy đủ format tiền tệ):

```bash
# Dự đoán cho tập dữ liệu bất kỳ
python predict.py --input data/raw/train.csv

# Tùy biến file đầu ra
python predict.py --input data/raw/train.csv --output output/KetQua_Moi.csv
```

Sử dụng trong mã Python / Notebook / Streamlit:

```python
from predict import predict
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
predictions = predict(df) # Trả về pd.Series giá nhà USD
print(predictions.head())
```
