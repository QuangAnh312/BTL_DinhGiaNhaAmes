# BTL Học máy – Dự đoán giá nhà Ames Housing

> Học phần: **Học máy (608807)** – Khoa CNTT, Trường ĐH Xây dựng Hà Nội · Lớp **69CS1**
> Đề tài: **Đề 3 – Định giá Bất động sản (Real Estate Valuation)**, dữ liệu Ames Housing
> Đề bài & rubric đầy đủ: [`docs/De_bai_va_Rubric_Bai_tap_lon.pdf`](docs/De_bai_va_Rubric_Bai_tap_lon.pdf)
> 👉 **Thành viên mới: đọc [`HUONG_DAN_BAT_DAU.md`](HUONG_DAN_BAT_DAU.md) trước tiên.**

---

## 1. Tổng quan bài toán

| Mục | Nội dung |
|---|---|
| Loại bài toán | Học có giám sát – **Hồi quy** |
| Dữ liệu | Kaggle *House Prices: Advanced Regression Techniques* – `train.csv` (1.460 căn × 81 cột). Nguồn gốc: De Cock (2011), *Journal of Statistics Education* |
| Biến đầu vào X | 79 đặc trưng (≈36 số, ≈43 phân loại): diện tích, chất lượng, vị trí, năm xây, số phòng, tiện ích… + đặc trưng tự tạo |
| Biến mục tiêu y | `SalePrice` (USD), huấn luyện trên `log1p(SalePrice)` |
| Chỉ số đánh giá | RMSE, MAE, R², RMSLE |
| Chia dữ liệu | 80% train / 20% test (`random_state=42`), **test chỉ dùng 1 lần ở cuối** |

**Đặc điểm dữ liệu cần xử lý:** nhiều giá trị khuyết (NA ở `PoolQC`, `Alley`, `Fence`, `GarageX`, `BsmtX`… nghĩa là *"không có"*); `SalePrice` lệch phải; ngoại lai `GrLivArea > 4000`.

**Mô hình dự kiến:** `DummyRegressor` (baseline) → Ridge/Lasso → KNN → Decision Tree → Random Forest → Gradient Boosting (HistGradientBoosting).

---

## 2. Yêu cầu bắt buộc (tóm tắt từ đề)

1. EDA & tiền xử lý: làm sạch, missing, ngoại lai, tương quan, chuẩn hóa, mã hóa.
2. Feature Engineering + phân tích mức độ quan trọng của đặc trưng.
3. ≥ 3 dòng thuật toán khác nhau (+ baseline).
4. `Pipeline` + `ColumnTransformer` – **chống rò rỉ dữ liệu**.
5. K-Fold CV + `GridSearchCV` / `RandomizedSearchCV`.
6. Bảng so sánh đa chỉ số, biểu đồ phần dư, giải thích vì sao mô hình tốt nhất vượt trội.
7. Xuất `best_model.joblib` + hàm dự đoán dữ liệu mới (`predict.py`).

### Thang điểm (rubric)

| Tiêu chí | Điểm | Để đạt Xuất sắc |
|---|---|---|
| EDA & Tiền xử lý | 2.0 | FE độc đáo, xử lý ngoại lai thông minh, code sạch |
| Pipeline & Thử nghiệm mô hình | 2.5 | Đa dạng tuyến tính → ensemble, Pipeline tự động hoàn toàn |
| CV & Tinh chỉnh | 2.0 | Learning/Validation Curve, đánh giá trên Test Set độc lập |
| Báo cáo & Mã nguồn | 1.5 | Báo cáo như bài báo, PEP8, README chuyên nghiệp, có `.joblib` |
| Thuyết trình & Vấn đáp | 2.0 | Demo Webapp (Streamlit), trả lời tốt phản biện |

### ⚠️ Quy định trừ điểm

- **Rò rỉ dữ liệu: −1.0 đến −2.0 điểm** (fit scaler/imputer trên toàn bộ dữ liệu, lộ test trong CV).
- **Đạo code / chép Kaggle-GitHub: cả nhóm 0 điểm.** Tham khảo ý tưởng được, **tự viết code**.
- **Điểm cá nhân = Điểm nhóm × Hệ số đóng góp ± Điểm vấn đáp.** Ai cũng phải hiểu **toàn bộ** code, không chỉ phần mình.

---

## 3. Cấu trúc thư mục

```
BTL_DinhGiaNhaAmes/
├── HUONG_DAN_BAT_DAU.md           # Hướng dẫn cài đặt & quy trình làm việc
├── docs/                          # Tài liệu nội bộ (đề bài, rubric, biên bản họp) – KHÔNG nộp
│   └── De_bai_va_Rubric_Bai_tap_lon.pdf
├── 01_Report/                     # Bao_cao_BTL_[TenNhom].pdf (15–25 trang)
├── 02_Slide/                      # Slide_Thuyet_trinh_[TenNhom].pdf (12–15 trang)
├── 03_SourceCode/
│   ├── data/
│   │   ├── raw/                   # train.csv (không commit, xem data/README.md)
│   │   ├── processed/             # train/test đã chia (nếu cần lưu)
│   │   └── README.md              # Hướng dẫn tải dữ liệu + trích dẫn nguồn
│   ├── notebooks/
│   │   ├── 01_EDA_Preprocessing.ipynb
│   │   ├── 02_Modeling_Evaluation.ipynb
│   │   └── drafts/                # Notebook nháp cá nhân (xóa trước khi nộp)
│   ├── src/                       # Code dùng chung: custom transformer, hàm build pipeline
│   ├── app/                       # Demo Streamlit (app.py)
│   ├── models/                    # best_model.joblib
│   ├── predict.py                 # Suy luận dữ liệu mới
│   ├── requirements.txt           # Phiên bản chính xác (pip freeze)
│   └── README.md                  # Hướng dẫn cài đặt & chạy
└── 04_Video_Demo/                 # Video demo 3 phút (khuyến khích)
```

> Khi nộp LMS: chép ra thư mục tên `[TenNhom]_69CS1_DinhGiaNhaAmes` theo quy định của đề, bỏ thư mục `docs/`.

---

## 4. Thành viên & phân công

| # | Họ tên | MSSV | Vai trò | Phụ trách chính | Chương báo cáo |
|---|---|---|---|---|---|
| 1 | **Ngô Quang Anh** | 0364469 | **Nhóm trưởng** | Quản lý tiến độ, repo & review code (kiểm tra leakage); chia train/test; **CV + RandomizedSearchCV**, Learning/Validation Curve; đánh giá cuối trên test | Ghép báo cáo, trang bìa + bảng phân công, Ch.4 (phần CV & tuning) |
| 2 | **Dương Hữu Hinh** | 0367869 | Kỹ thuật chính | **Kiến trúc Pipeline + ColumnTransformer**, custom transformer cho FE/missing (`src/`), huấn luyện ≥5 mô hình, xuất `best_model.joblib`, lõi `predict.py` | Ch.3 (sơ đồ Pipeline) |
| 3 | Lưu Ngọc Tuấn Anh | 1644069 | Dữ liệu | EDA (phân phối, tương quan, missing, ngoại lai), chiến lược tiền xử lý → notebook 01 | Ch.1, Ch.2 |
| 4 | Dương Hoàng Anh | 0364169 | Phân tích | Đề xuất đặc trưng mới; bảng so sánh mô hình, **phân tích lỗi** (residual plot), feature importance / permutation importance | Ch.3 (nguyên lý toán), Ch.4 (phân tích lỗi) |
| 5 | Phạm Tiến Quang | 0371969 | Sản phẩm | **Demo Streamlit**, `requirements.txt`, README mã nguồn, slide, video demo | Ch.5, Tài liệu tham khảo, Slide |

- Hinh phụ trách phần khó nhất (Pipeline, 2.5đ); Quang Anh phụ trách phần khó thứ hai (CV & tuning, 2đ) và cùng review với Hinh.
- Tỷ lệ đóng góp mặc định 20%/người, chốt lại trước khi nộp dựa trên commit + công việc thực tế, **cả nhóm ký xác nhận**.
- Mỗi người phải tự chạy lại được 2 notebook và giải thích được mọi bước để chuẩn bị vấn đáp.
- Tài liệu tham khảo: ai dùng nguồn nào (bài báo, tài liệu scikit-learn…) thì gửi Tiến Quang ghi vào danh sách chung ngay khi dùng.
- Đề xuất đặc trưng và bảng nhóm biến của Hoàng Anh: [tài liệu bàn giao](01_Report/HoangAnh/De_xuat_dac_trung_va_nhom_bien.md), [notebook nháp](03_SourceCode/notebooks/drafts/03_Feature_Proposal_HoangAnh.ipynb).

---

## 5. Lịch trình (theo milestone của đề)

Lớp học **thứ 4 hằng tuần**: Buổi 2 = 18/11 · Buổi 3 = 25/11 · Buổi 4 = 02/12 · Buổi 5 = 09/12 · Buổi 6 = 16/12 · Buổi 7 = 23/12 · Buổi 8 = 30/12.

### Giai đoạn 1 – Hoàn thành yêu cầu Buổi 4 sớm: **hạn nội bộ T6 27/11**

| Ngày | Việc | Người |
|---|---|---|
| T6–T7 20–21/11 | Tạo repo GitHub, mời thành viên; tải `train.csv`; chia train/test cố định (lưu index) | Quang Anh |
| T6–CN 20–22/11 | EDA: tổng quan, phân phối `SalePrice` (log), missing (NA = "không có" hay khuyết thật), ngoại lai | Tuấn Anh |
| T7–T2 21–23/11 | Khung `Pipeline` + `ColumnTransformer` (num: impute + scale; cat: impute + one-hot) | Hinh |
| T7–T2 21–23/11 | Đề xuất đặc trưng mới (TotalSF, HouseAge, TotalBath…), soạn bảng nhóm biến | Hoàng Anh |
| T2–T3 23–24/11 | EDA: tương quan, biểu đồ theo `Neighborhood`/`OverallQual`; chốt chiến lược tiền xử lý | Tuấn Anh |
| **T4 25/11 (Buổi 3)** | Họp trên lớp: review EDA, chốt chiến lược tiền xử lý → [biên bản + việc phát sinh](docs/Bien_ban_hop_25_11.md) | Cả nhóm |
| T4–T5 25–26/11 | Baseline `DummyRegressor` + `Ridge` trong Pipeline, 5-Fold CV, bảng RMSE/MAE/R²/RMSLE | Hinh |
| T4–T5 25–26/11 | Báo cáo tiến độ 01 (sơ đồ EDA + chiến lược tiền xử lý, 3–5 trang) | Tiến Quang (tổng hợp từ Tuấn Anh, Hinh) |
| **T6 27/11** | Review leakage, notebook chạy *Restart & Run All*, merge `main` | Quang Anh |
| T4 02/12 (Buổi 4) | **Checkpoint 1** – GV kiểm tra | Cả nhóm |

### Giai đoạn 2 – Hoàn thiện sản phẩm

| Mốc | Hạn | Việc cần xong | Người chính |
|---|---|---|---|
| — | 28/11 – 09/12 | Đủ ≥5 mô hình; FE trong Pipeline (custom transformer); bắt đầu tuning | Hinh, Hoàng Anh, Quang Anh |
| **Hạn nội bộ** | **CN 13/12** | Pipeline hoàn chỉnh (FE + tiền xử lý + mô hình), tuning xong, xuất bản `best_model.joblib` đầu tiên, `predict.py` chạy được | Hinh, Quang Anh |
| — | 13/12 – 16/12 | Learning/Validation Curve, phân tích lỗi, Streamlit; bắt đầu viết báo cáo | Cả nhóm |
| **Checkpoint 2** | **T4 16/12 (Buổi 6)** | Theo đề: *Hoàn thiện Pipeline & Nộp sản phẩm*. Mang bản chạy được lên lớp cho GV kiểm tra | Cả nhóm |
| — | 16/12 – 20/12 | Sửa theo góp ý GV; hoàn thiện báo cáo, slide, video demo | Cả nhóm |
| **Hạn nội bộ** | **CN 20/12** | Báo cáo + slide + code hoàn chỉnh, mọi người chạy thử trên máy mình | Cả nhóm |
| **M2** | **T3 22/12 (trước 24h Buổi 7)** | Nộp LMS: PDF báo cáo, slide, source code, `.joblib` | Quang Anh |
| **M3** | 23/12 & 30/12 | Thuyết trình, live demo, vấn đáp cá nhân | Cả nhóm |

---

## 6. Quy tắc làm việc

**Kỹ thuật (bắt buộc):**
- Chia train/test **một lần duy nhất**, `random_state=42`, trước mọi bước fit.
- **Mọi** bước học từ dữ liệu (impute, scale, encode, tạo đặc trưng, chọn đặc trưng) phải nằm **trong Pipeline**. Chỉ bỏ ngoại lai trên tập train, trước khi đưa vào Pipeline.
- Custom transformer đặt trong `03_SourceCode/src/` (không định nghĩa trong notebook), nếu không `predict.py` sẽ không load được `.joblib`.
- Code theo PEP8 (`black` + `flake8`), comment tiếng Việt cho các bước quan trọng.
- Không commit dữ liệu gốc, file lớn, hoặc `CLAUDE.md`.

**Git:**
- Nhánh `main` luôn chạy được. Mỗi người làm trên nhánh riêng (`eda`, `pipeline`, `tuning`, `demo`…) rồi tạo Pull Request.
- Nhóm trưởng (hoặc Hinh) review trước khi merge.
- Commit message rõ ràng: `eda: thêm biểu đồ tương quan`, `pipeline: thêm custom transformer TotalSF`.
- Notebook: *Restart & Run All* trước khi commit để output đúng thứ tự.

---

## 7. Cài đặt nhanh

```bash
git clone https://github.com/QuangAnh312/BTL_DinhGiaNhaAmes.git
cd BTL_DinhGiaNhaAmes/03_SourceCode
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Tải train.csv vào data/raw/ (xem data/README.md)
jupyter notebook notebooks/
streamlit run app/app.py
python predict.py --input <file.csv>
```

---

## 8. Checklist trước khi nộp

- [ ] 2 notebook *Restart & Run All* không lỗi, kết quả trùng báo cáo
- [ ] Không có bước `fit` nào trên toàn bộ dữ liệu / tập test
- [ ] Có baseline, ≥3 dòng thuật toán, bảng so sánh RMSE/MAE/R²/RMSLE
- [ ] Có Learning Curve / Validation Curve, residual plot, feature importance
- [ ] `best_model.joblib` + `predict.py` chạy trên máy sạch
- [ ] `requirements.txt` ghi phiên bản chính xác
- [ ] Báo cáo 15–25 trang, Times New Roman 13, giãn dòng 1.3–1.5, bảng phân công + % + chữ ký
- [ ] MSSV trên trang bìa đã được từng người tự kiểm tra lại
- [ ] Ch.4: ghi rõ bài hồi quy nên dùng Residual Plot thay cho ROC-AUC (ROC-AUC chỉ dành cho phân loại)
- [ ] Trích dẫn nguồn dữ liệu và tài liệu tham khảo
- [ ] Gói nộp LMS có kèm `data/raw/train.csv` để GV chạy lại notebook ngay (repo GitHub thì không commit)
- [ ] Slide 12–15 trang, video demo dự phòng
- [ ] Đặt tên thư mục đúng `[TenNhom]_69CS1_[TenDeTai]`
