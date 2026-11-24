"""Đọc dữ liệu và chia train/test cố định dùng chung cho cả nhóm.

Cách chia: train_test_split(test_size=0.2, random_state=42) chạy MỘT lần,
danh sách Id của tập test được lưu vào data/processed/test_ids.csv và commit.
Mọi notebook đều gọi load_split() nên ai cũng dùng đúng một tập test.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "SalePrice"

# Tính đường dẫn từ vị trí file này để chạy từ notebooks/ hay từ 03_SourceCode/
# đều đúng (không phụ thuộc thư mục đang đứng).
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "train.csv"
TEST_IDS_PATH = ROOT / "data" / "processed" / "test_ids.csv"


def load_raw(path=RAW_PATH):
    """Đọc file train.csv gốc của Kaggle (1.460 dòng x 81 cột).

    Mặc định pandas coi chuỗi "None" là NaN, làm 864 nhà có MasVnrType = "None"
    (không ốp đá, là một giá trị thật) bị gộp với 8 nhà thiếu dữ liệu.
    Vì vậy chỉ coi "NA" và ô trống là thiếu.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=["NA", ""])


def make_split(df, path=TEST_IDS_PATH):
    """Chia 80/20 và lưu Id tập test. Chỉ chạy một lần (nhóm trưởng)."""
    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    test_ids = test_df[["Id"]].sort_values("Id")
    test_ids.to_csv(path, index=False)
    return test_ids


def load_split(raw_path=RAW_PATH, ids_path=TEST_IDS_PATH):
    """Trả về X_train, X_test, y_train, y_test theo danh sách Id đã lưu.

    Cột Id bị bỏ khỏi X vì chỉ là số thứ tự, không mang thông tin về giá.
    """
    df = load_raw(raw_path)
    test_ids = pd.read_csv(ids_path)["Id"]
    is_test = df["Id"].isin(test_ids)

    train_df = df[~is_test]
    test_df = df[is_test]

    X_train = train_df.drop(columns=[TARGET, "Id"])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET, "Id"])
    y_test = test_df[TARGET]
    return X_train, X_test, y_train, y_test


def remove_outliers(X_train, y_train, max_area=4000):
    """Bỏ nhà có diện tích ở (GrLivArea) > max_area feet vuông.

    Chỉ gọi trên tập TRAIN, trước khi đưa vào Pipeline; tập test giữ nguyên
    để đánh giá trung thực. Ngưỡng 4000 theo khuyến nghị của tác giả bộ dữ liệu
    (De Cock, 2011): các nhà này rất hiếm và vài căn là giao dịch bất thường
    (diện tích rất lớn nhưng giá thấp), làm lệch mô hình tuyến tính.
    """
    keep = X_train["GrLivArea"] <= max_area
    return X_train[keep], y_train[keep]


if __name__ == "__main__":
    # Chạy: python -m src.data (trong 03_SourceCode/)
    # Không ghi đè nếu đã có file, để tập test không bị đổi giữa chừng.
    # Thông báo viết không dấu vì terminal Windows (cp1252) không in được tiếng Việt.
    if TEST_IDS_PATH.exists():
        print(f"Da co {TEST_IDS_PATH.name}, khong chia lai.")
    else:
        ids = make_split(load_raw())
        print(f"Da luu {len(ids)} Id tap test vao {TEST_IDS_PATH}")
