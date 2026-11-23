# Hướng dẫn trước khi bắt đầu

> Mọi thành viên làm xong **Bước 1–5 trước T7 21/11** và báo lên nhóm.

## Bước 1. Cài công cụ (dùng chung phiên bản để tránh lỗi "máy tao chạy được")

| Công cụ | Ghi chú |
|---|---|
| **Python 3.11** | Cả nhóm dùng cùng bản. Windows: tick *Add Python to PATH* |
| **Git** | https://git-scm.com |
| **VS Code** + extension *Python*, *Jupyter* | Hoặc Jupyter Notebook |
| Tài khoản **GitHub** và **Kaggle** | Gửi username GitHub cho nhóm trưởng để được mời vào repo |

## Bước 2. Đọc trước (khoảng 30 phút)

1. `README.md`: nắm bài toán, phân công, lịch, quy tắc.
2. `docs/De_bai_va_Rubric_Bai_tap_lon.pdf`: đặc biệt mục **Rubric** và **Quy định trừ điểm**.
3. `data_description.txt` (tải ở Bước 4): ý nghĩa các cột.

## Bước 3. Lấy code về máy

```bash
git config --global user.name "Ho Ten"
git config --global user.email "email@github"
git clone https://github.com/QuangAnh312/BTL_DinhGiaNhaAmes.git
cd BTL_DinhGiaNhaAmes/03_SourceCode
```

## Bước 4. Tạo môi trường + tải dữ liệu

```bash
python --version                  # phải là 3.11.x
python -m venv .venv              # máy có nhiều bản Python: py -3.11 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- **Phải nâng cấp pip trước khi cài.** pip 24.0 đi kèm Python 3.11 đọc `requirements.txt`
  bằng bảng mã Windows nên lỗi `UnicodeDecodeError` (do file có comment tiếng Việt).
- Trong VS Code chọn kernel `.venv` khi mở notebook. Nếu xóa/tạo lại `.venv` thì phải chọn lại kernel.
- Tải `train.csv` và `data_description.txt` theo `data/README.md`, đặt vào `data/raw/`.

## Bước 5. Kiểm tra

```bash
python -c "import sklearn, pandas; print(sklearn.__version__, pandas.__version__)"
python -c "import pandas as pd; print(pd.read_csv('data/raw/train.csv').shape)"   # (1460, 81)
```
Chạy được thì nhắn nhóm "OK", lỗi thì gửi ảnh chụp lỗi lên nhóm.

---

## Quy trình làm việc mỗi ngày

```bash
git checkout main && git pull                 # 1. cập nhật code mới nhất
git checkout -b <ten-nhanh>                   # 2. tạo nhánh, vd: eda-missing, pipeline-baseline
# ... làm việc ...
# 3. Notebook: Kernel > Restart & Run All, không lỗi mới commit
git add <file> && git commit -m "eda: phân tích missing values"
git push -u origin <ten-nhanh>                # 4. đẩy lên
# 5. Mở Pull Request vào main trên GitHub, nhờ Quang Anh review
```

**Tránh xung đột notebook** (Git rất khó gộp file `.ipynb`):
- Mỗi notebook chính chỉ **một người sửa**: `01_EDA_Preprocessing` là Tuấn Anh, `02_Modeling_Evaluation` là Hinh.
- Người khác thử nghiệm trong `notebooks/drafts/<ten>_<viec>.ipynb`, xong thì gửi người phụ trách gộp vào. Thư mục `drafts/` sẽ xóa trước khi nộp.
- Hàm/class dùng chung viết vào `src/` (file `.py` gộp dễ hơn).

**Không bao giờ:** commit thẳng lên `main`, commit `data/raw/`, dùng `git push --force`.

---

## Kiến thức cần nắm trước khi code

Ai cũng phải hiểu, vì có vấn đáp cá nhân:
- `Pipeline`, `ColumnTransformer`: https://scikit-learn.org/stable/modules/compose.html
- Rò rỉ dữ liệu và lỗi thường gặp: https://scikit-learn.org/stable/common_pitfalls.html
- Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Tinh chỉnh siêu tham số: https://scikit-learn.org/stable/modules/grid_search.html

**Tự kiểm tra:** giải thích được vì sao `scaler.fit(X)` trên toàn bộ dữ liệu trước khi chia train/test là sai.

---

