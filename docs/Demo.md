# 1. Cấu trúc mới của thư mục `03_SourceCode/`

```text
03_SourceCode/
├── train/                                  # [MỚI] Thư mục chứa code huấn luyện mô hình
│   ├── train_demo.py                       # So sánh 7 mô hình qua 5-Fold CV & xuất joblib
│   └── train_all_models.py                 # Huấn luyện toàn bộ 7 mô hình, xuất CSV và đồ thị
├── output/                                 # [MỚI] Thư mục chứa các file kết quả và sơ đồ
│   ├── GiaNha_Dummy.csv                    # Kết quả dự đoán mô hình Dummy Baseline
│   ├── GiaNha_Ridge.csv                    # Kết quả dự đoán mô hình Ridge Regression
│   ├── GiaNha_Lasso.csv                    # Kết quả dự đoán mô hình Lasso Regression (Tối ưu)
│   ├── GiaNha_KNN.csv                      # Kết quả dự đoán mô hình K-Nearest Neighbors
│   ├── GiaNha_DecisionTree.csv             # Kết quả dự đoán mô hình Decision Tree
│   ├── GiaNha_RandomForest.csv             # Kết quả dự đoán mô hình Random Forest
│   ├── GiaNha_HistGB.csv                   # Kết quả dự đoán mô hình HistGradientBoosting
│   ├── GiaNha_DuDoan.csv                   # Kết quả suy luận dữ liệu mới từ predict.py
│   └── plots/                              # Sơ đồ đánh giá 4 góc trực quan (.png)
│       ├── DanhGia_Dummy.png
│       ├── DanhGia_Ridge.png
│       ├── DanhGia_Lasso.png
│       ├── DanhGia_KNN.png
│       ├── DanhGia_DecisionTree.png
│       ├── DanhGia_RandomForest.png
│       └── DanhGia_HistGB.png
├── notebooks/                              # Thư mục Notebooks
│   ├── 01_EDA_Preprocessing.ipynb          # Notebook khám phá & tiền xử lý EDA
│   ├── 02_Modeling_Evaluation.ipynb         # Notebook mô hình hóa tổng hợp
│   └── train_models/                       # [MỚI] 7 Notebook huấn luyện riêng từng mô hình
│       ├── 01_Train_Dummy.ipynb
│       ├── 02_Train_Ridge.ipynb
│       ├── 03_Train_Lasso.ipynb
│       ├── 04_Train_KNN.ipynb
│       ├── 05_Train_DecisionTree.ipynb
│       ├── 06_Train_RandomForest.ipynb
│       └── 07_Train_HistGB.ipynb
├── train_notebooks -> notebooks/train_models # Phím tắt liên kết nhanh đến các notebook train
├── data/
│   ├── raw/train.csv                       # Tập dữ liệu gốc 1460 dòng
│   └── processed/test_ids.csv              # Danh sách 292 ID tập test cố định của nhóm
├── models/
│   └── best_model.joblib                   # Mô hình tối ưu đóng gói (Lasso)
├── src/                                    # Thư viện dùng chung (Pipeline & Transformers)
├── predict.py                              # CLI suy luận (mặc định xuất vào output/)
└── README.md                               # Hướng dẫn chi tiết đã cập nhật
```

*(Lưu ý: Các file `train_demo.py` và `du_doan.csv` ở ngoài thư mục gốc `03_SourceCode/` đã được chuyển vào đúng vị trí và **xóa bỏ hoàn toàn ở ngoài**).*

---

# 2. Định dạng mới của các file output giá nhà (`GiaNha_<MôHình>.csv`)

Toàn bộ các file trong thư mục `output/` đã được chuẩn hóa lại gồm 6 cột:
* **`Id`**: Mã căn nhà trong bộ dữ liệu.
* **`GiaNha_ThucTe`**: Giá bán thực tế trong tập test ($).
* **`GiaNha_DuDoan`**: Giá bán mô hình dự đoán (làm tròn 2 chữ số thập phân).
* **`GiaNha_DinhDang`**: Giá tiền đã được format dạng tiền tệ USD đẹp mắt (`$132,220.67`).
* **`ChenhLech_USD`**: Sai số tuyệt đối $|Y_{thực tế} - Y_{dự đoán}|$ theo USD.
* **`TiLe_SaiLech_%`**: Tỉ lệ phần trăm sai lệch tương đối.

**Minh họa 5 dòng đầu file `GiaNha_Lasso.csv`:**
```csv
Id,GiaNha_ThucTe,GiaNha_DuDoan,GiaNha_DinhDang,ChenhLech_USD,TiLe_SaiLech_%
16,132000.00,132220.67,"$132,220.67",220.67,0.17%
24,129900.00,144210.71,"$144,210.71",14310.71,11.02%
30,68500.00,72042.57,"$72,042.57",3542.57,5.17%
33,179900.00,191184.29,"$191,184.29",11284.29,6.27%
44,130250.00,131880.00,"$131,880.00",1630.00,1.25%
```

---

# 3. Folder các file `.ipynb` huấn luyện thử từng mô hình

Các notebook huấn luyện độc lập đặt tại `notebooks/train_models/` (hoặc truy cập qua phím tắt `train_notebooks/`):

1. **`01_Train_Dummy.ipynb`**: Mô hình Baseline cơ sở.
2. **`02_Train_Ridge.ipynb`**: Hồi quy tuyến tính Ridge (Điều chuẩn L2).
3. **`03_Train_Lasso.ipynb`**: Hồi quy Lasso (Điều chuẩn L1 - Mô hình tối ưu).
4. **`04_Train_KNN.ipynb`**: Thuật toán K-Nearest Neighbors.
5. **`05_Train_DecisionTree.ipynb`**: Mô hình Cây quyết định CART.
6. **`06_Train_RandomForest.ipynb`**: Rừng ngẫu nhiên đóng bao Ensemble.
7. **`07_Train_HistGB.ipynb`**: HistGradientBoosting tương tự LightGBM.

**Trong mỗi notebook đều được cấu trúc chuẩn 7 bước:**
- Nạp thư viện và cấu hình đường dẫn tương thích.
- Nạp tập train/test sạch qua `load_split()` từ `src.data`.
- Khởi tạo Pipeline 5 bước tương ứng với mô hình đó.
- Kiểm định chéo 5-Fold CV trên tập train.
- Đánh giá trên tập test độc lập (292 căn).
- **Vẽ sơ đồ đánh giá chuyên sâu 4 góc** (Actual vs Predicted, Residuals Scatter, Residuals Distribution, Thẻ tổng kết Metrics).
- Xuất dữ liệu dự đoán format chuẩn vào `output/GiaNha_<MôHình>.csv`.

---

# 4. Sơ đồ đánh giá trực quan từng mô hình (`output/plots/`)

Mỗi mô hình đã được vẽ và lưu biểu đồ đánh giá chất lượng tại `output/plots/`:

* `DanhGia_Lasso.png`
* `DanhGia_Ridge.png`
* `DanhGia_HistGB.png`
* `DanhGia_RandomForest.png`
* `DanhGia_KNN.png`
* `DanhGia_DecisionTree.png`
* `DanhGia_Dummy.png`

---

# 5. Hướng dẫn các câu lệnh chạy mới

Bạn có thể chạy các lệnh này từ thư mục `03_SourceCode` hoặc từ thư mục gốc dự án:

```bash
cd 03_SourceCode

# 1. Chạy so sánh 7 mô hình qua 5-Fold CV và lưu best_model.joblib:
python train/train_demo.py

# 2. Huấn luyện toàn bộ 7 mô hình, xuất 7 file CSV và 7 sơ đồ đánh giá:
python train/train_all_models.py

# 3. Dự đoán dữ liệu mới (kết quả tự động lưu vào output/GiaNha_DuDoan.csv):
python predict.py --input data/raw/train.csv
```
