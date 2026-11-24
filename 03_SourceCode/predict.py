"""Suy luận giá nhà cho dữ liệu mới bằng mô hình tốt nhất (models/best_model.joblib).

Chạy từ thư mục bất kỳ hoặc từ 03_SourceCode/:
    python predict.py --input data/raw/train.csv --output du_doan.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Union

import joblib
import pandas as pd

# Đảm bảo nạp được package src dù đứng ở bất kỳ vị trí nào
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Cần import để joblib nhận diện được custom transformers
import src.transformers  # noqa: E402, F401

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Nạp mô hình đã đóng gói từ file joblib."""
    path = Path(model_path)
    if not path.exists():
        msg = (
            f"Không tìm thấy mô hình tại '{path}'. "
            "Vui lòng chạy 'python train/train_demo.py' trước để huấn luyện."
        )
        raise FileNotFoundError(msg)
    return joblib.load(path)


def predict(
    data: Union[pd.DataFrame, dict, list],
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> pd.Series:
    """Dự đoán giá nhà từ dữ liệu thô (DataFrame hoặc Dict), trả về USD."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Dữ liệu đầu vào phải là DataFrame, dict hoặc list các dict.")

    model = load_model(model_path)
    X = df.drop(columns=["Id", "SalePrice"], errors="ignore")
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="SalePrice_pred")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dự đoán giá nhà Ames Housing bằng mô hình tối ưu"
    )
    parser.add_argument(
        "--input", required=True, help="Đường dẫn file CSV dữ liệu đầu vào"
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Đường dẫn lưu file CSV kết quả dự đoán (mặc định: predictions.csv)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Đường dẫn file mô hình .joblib",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[LỖI] File đầu vào không tồn tại: {input_path}")
        sys.exit(1)

    print(f"[*] Đang đọc file đầu vào: {input_path}")
    df = pd.read_csv(input_path, keep_default_na=False, na_values=["NA", ""])

    try:
        preds = predict(df, model_path=args.model)
    except FileNotFoundError as err:
        print(f"[LỖI] {err}")
        sys.exit(1)

    out_df = pd.DataFrame()
    if "Id" in df.columns:
        out_df["Id"] = df["Id"]
    out_df["SalePrice_pred"] = preds.round(2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[✓] Đã lưu {len(out_df)} kết quả dự đoán vào: {output_path}")
    print("\nMột số dự đoán đầu tiên:")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
