"""Script huấn luyện và so sánh 7 mô hình bằng 5-Fold Cross Validation.

Vị trí: 03_SourceCode/train/train_demo.py
Chạy từ bất kỳ đâu:
    python train/train_demo.py                # Ordinal encoding (chuẩn)
    python train/train_demo.py --no-ordinal  # One-hot encoding toàn bộ
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

# Thư mục gốc 03_SourceCode/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import load_split, remove_outliers  # noqa: E402
from src.pipeline import RANDOM_STATE, build_pipeline, get_models  # noqa: E402

DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

SCORING = {
    "rmse": "neg_root_mean_squared_error",  # USD
    "mae": "neg_mean_absolute_error",  # USD
    "r2": "r2",
    "rmsle": "neg_root_mean_squared_log_error",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện và so sánh các mô hình định giá nhà Ames"
    )
    parser.add_argument(
        "--data", default=str(DATA_PATH), help="Đường dẫn đến file train.csv"
    )
    parser.add_argument(
        "--output-model",
        default=str(DEFAULT_MODEL_PATH),
        help="Đường dẫn lưu best_model.joblib",
    )
    parser.add_argument(
        "--no-ordinal",
        action="store_true",
        help="Sử dụng One-Hot thay vì Ordinal Encoding cho các cột chất lượng",
    )
    args = parser.parse_args()

    ordinal = not args.no_ordinal
    data_file = Path(args.data)
    model_output = Path(args.output_model)

    if not data_file.exists():
        print(f"[LỖI] Không tìm thấy file dữ liệu tại: {data_file}")
        print("Vui lòng đặt train.csv vào 'data/raw/train.csv' hoặc dùng --data")
        sys.exit(1)

    print(f"[*] Đang nạp dữ liệu từ: {data_file}")
    X_train, _, y_train, _ = load_split(raw_path=data_file)
    n_before = len(X_train)
    X_train, y_train = remove_outliers(X_train, y_train)
    print(
        f"[*] Đã loại {n_before - len(X_train)} mẫu ngoại lai. "
        f"Train còn: {len(X_train)} dòng."
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    rows = {}

    print(f"[*] Bắt đầu kiểm định chéo 5-Fold CV (chế độ ordinal={ordinal}):")
    for name, model in models.items():
        pipe = build_pipeline(model, ordinal=ordinal)
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        rows[name] = {k: res[f"test_{k}"].mean() for k in SCORING}
        rows[name]["rmse"] *= -1
        rows[name]["mae"] *= -1
        rows[name]["rmsle"] *= -1
        print(
            f"    - Hoàn thành {name:<14} | "
            f"RMSLE: {rows[name]['rmsle']:.4f} | R²: {rows[name]['r2']:.4f}"
        )

    table = pd.DataFrame(rows).T.sort_values("rmsle")
    print("\n" + "=" * 65)
    print("BẢNG KẾT QUẢ SO SÁNH 7 MÔ HÌNH (SẮP XẾP THEO RMSLE TĂNG DẦN):")
    print("=" * 65)
    print(table.round(4).to_string())
    print("=" * 65)

    # Fit mô hình tốt nhất trên toàn bộ train và lưu file joblib
    best_name = table.index[0]
    best_rmsle = table.loc[best_name, "rmsle"]
    print(f"\n[+] Mô hình tối ưu: {best_name} (RMSLE = {best_rmsle:.4f})")
    print(f"[*] Đang huấn luyện lại {best_name} trên toàn bộ tập train...")
    final_pipeline = build_pipeline(models[best_name], ordinal=ordinal)
    final_pipeline.fit(X_train, y_train)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_output)
    print(f"[✓] Đã lưu mô hình tốt nhất thành công vào: {model_output}")


if __name__ == "__main__":
    main()
