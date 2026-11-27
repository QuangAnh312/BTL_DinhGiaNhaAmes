# Demo phần Pipeline và mô hình (cập nhật 26/11)

Pipeline đã cài theo chiến lược tiền xử lý chốt ngày 25/11 (`docs/Bien_ban_hop_25_11.md`, Mục 4).

## 1. Các file liên quan trong `03_SourceCode/`

```text
03_SourceCode/
├── src/
│   ├── data.py              # load_split(): chia train/test cố định theo test_ids.csv; remove_outliers()
│   ├── transformers.py      # MissingHandler, OrdinalMapper, FeatureAdder
│   ├── pipeline.py          # build_preprocessor(), build_pipeline(), get_models()
│   └── plots.py             # plot_model_evaluation(): biểu đồ đánh giá 4 góc
├── train/
│   └── train_demo.py        # So sánh 7 mô hình bằng 5-Fold CV, lưu mô hình tốt nhất
├── notebooks/
│   └── 02_Modeling_Evaluation.ipynb   # Pipeline, CV 7 mô hình, đối chứng ordinal/one-hot
├── models/
│   └── best_model.joblib    # Mô hình tốt nhất hiện tại (Lasso, alpha=0.001)
└── predict.py               # Dự đoán giá cho file CSV mới
```

## 2. Pipeline

`TransformedTargetRegressor(log1p/expm1)` bọc `Pipeline` 5 bước:

1. `MissingHandler`: NA "không có" → `"None"`/0; `GarageYrBlt` ← `YearBuilt`; `LotFrontage` ← trung vị theo
   `Neighborhood`; `Electrical`, `BsmtExposure`, `BsmtFinType2` ← mode (học từ fold train).
2. `OrdinalMapper`: 12 cột thứ bậc → số (`"passthrough"` khi `ordinal=False`). `BsmtFinType1/2` đi vào one-hot.
3. `FeatureAdder`: `TotalSF`, `HouseAge`, `RemodAge`, `TotalBath`, `TotalPorchSF`, `HasPool`, `Has2ndFloor`,
   `HasGarage`, `HasFireplace`; `MSSubClass`/`MoSold`/`YrSold` → chuỗi; bỏ `PoolQC`.
4. `ColumnTransformer` 3 nhánh: `num` (median → chuẩn hóa), `cat` (`"None"` → one-hot, `min_frequency=0.01`),
   `log` (8 biến lệch: median → `log1p` → chuẩn hóa).
5. Mô hình hồi quy.

Bỏ 3 căn `GrLivArea > 4000` chỉ trên train, ngoài Pipeline (`remove_outliers()`).

## 3. Kết quả 5-Fold CV trên train (1.165 căn, RMSLE)

| Mô hình | RMSLE | ± std |
|---|---|---|
| Lasso | 0,1086 | 0,0157 |
| Ridge | 0,1101 | 0,0133 |
| HistGB | 0,1223 | 0,0140 |
| RandomForest | 0,1330 | 0,0133 |
| KNN | 0,1570 | 0,0089 |
| DecisionTree | 0,1996 | 0,0102 |
| Dummy | 0,3887 | 0,0057 |

Số lấy từ `python train/train_demo.py` (Windows, Python 3.11). Đủ 4 chỉ số RMSE/MAE/R²/RMSLE xem notebook 02,
Mục 3. Chưa tuning; tập test chưa dùng (đánh giá một lần ở cuối, notebook 02 Mục 5).

## 4. Lệnh chạy (trong `03_SourceCode/`)

```bash
# So sánh 7 mô hình bằng 5-Fold CV, lưu models/best_model.joblib
python train/train_demo.py

# Đối chứng: one-hot cho cả 12 cột thứ bậc
python train/train_demo.py --no-ordinal

# Dự đoán cho file CSV mới (mặc định lưu predictions.csv, cột Id + SalePrice_pred)
python predict.py --input duong_dan_file.csv --output predictions.csv
```

Dùng trong Python:

```python
import pandas as pd
from predict import predict

df = pd.read_csv("duong_dan_file.csv", keep_default_na=False, na_values=["NA", ""])
gia = predict(df)  # pd.Series giá nhà (USD)
```
