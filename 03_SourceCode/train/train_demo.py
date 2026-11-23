"""Script huấn luyện và so sánh 7 mô hình bằng 5-Fold Cross Validation.

Vị trí: 03_SourceCode/train/train_demo.py
Chạy từ bất kỳ đâu:
    python train/train_demo.py                # Ordinal encoding (chuẩn)
    python train/train_demo.py --no-ordinal  # One-hot encoding toàn bộ
    python train/train_demo.py --outlier-mode de_cock
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, train_test_split

# Thư mục gốc 03_SourceCode/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import RANDOM_STATE, build_pipeline, get_models

DATA_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
TEST_IDS_PATH = ROOT_DIR / "data" / "processed" / "test_ids.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"

SCORING = {
    "rmse": "neg_root_mean_squared_error",  # USD
    "mae": "neg_mean_absolute_error",  # USD
    "r2": "r2",
    "rmsle": "neg_root_mean_squared_log_error",
}


def load_dataset(data_path: Path, test_ids_path: Path):
    """Nạp dữ liệu và tách tập train/test đồng bộ với quy ước chung của nhóm."""
    df = pd.read_csv(data_path)
    if test_ids_path.exists() and "Id" in df.columns:
        test_ids = pd.read_csv(test_ids_path)["Id"]
        is_test = df["Id"].isin(test_ids)
        train_df = df[~is_test]
        test_df = df[is_test]
        X_train = train_df.drop(columns=["SalePrice", "Id"], errors="ignore")
        y_train = train_df["SalePrice"]
        X_test = test_df.drop(columns=["SalePrice", "Id"], errors="ignore")
        y_test = test_df["SalePrice"]
    else:
        df_clean = df.drop(columns=["Id"], errors="ignore")
        X, y = df_clean.drop(columns=["SalePrice"]), df_clean["SalePrice"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện và so sánh các mô hình định giá nhà Ames")
    parser.add_argument(
        "--data", default=str(DATA_PATH), help="Đường dẫn đến file train.csv"
    )
    parser.add_argument(
        "--output-model", default=str(DEFAULT_MODEL_PATH), help="Đường dẫn lưu best_model.joblib"
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu kết quả dự đoán"
    )
    parser.add_argument(
        "--no-ordinal",
        action="store_true",
        help="Sử dụng One-Hot thay vì Ordinal Encoding cho các cột chất lượng",
    )
    parser.add_argument(
        "--outlier-mode",
        choices=["strict", "de_cock"],
        default="strict",
        help="Chế độ lọc ngoại lai: 'strict' (GrLivArea <= 4000) hoặc 'de_cock' (GrLivArea > 4000 & SalePrice < 300k)",
    )
    args = parser.parse_args()

    ordinal = not args.no_ordinal
    data_file = Path(args.data)
    model_output = Path(args.output_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        print(f"[LỖI] Không tìm thấy file dữ liệu tại: {data_file}")
        print("Vui lòng đặt train.csv vào 'data/raw/train.csv' hoặc dùng cờ --data <đường_dẫn>")
        sys.exit(1)

    print(f"[*] Đang nạp dữ liệu từ: {data_file}")
    X_train, X_test, y_train, y_test = load_dataset(data_file, TEST_IDS_PATH)
    print(f"[*] Kích thước dữ liệu gốc: X_train={X_train.shape}, X_test={X_test.shape}")

    # Xử lý ngoại lai chỉ trên tập train (giữ nguyên tập test để đánh giá khách quan)
    if args.outlier_mode == "strict":
        keep = X_train["GrLivArea"] <= 4000
    else:
        keep = ~((X_train["GrLivArea"] > 4000) & (y_train < 300000))

    removed_count = len(X_train) - keep.sum()
    X_train, y_train = X_train[keep], y_train[keep]
    print(f"[*] Đã loại {removed_count} mẫu ngoại lai (chế độ: {args.outlier_mode}). Tập train còn: {len(X_train)} dòng.")

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
        print(f"    - Hoàn thành {name:<14} | RMSLE: {rows[name]['rmsle']:.4f} | R²: {rows[name]['r2']:.4f}")

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

    # Xuất file dự đoán đã format vào thư mục output/
    y_pred = final_pipeline.predict(X_test)
    test_df_ids = pd.read_csv(TEST_IDS_PATH)["Id"].values if TEST_IDS_PATH.exists() else X_test.index
    diff = (y_test.values - y_pred)
    abs_diff = abs(diff)
    pct_diff = (abs_diff / y_test.values) * 100

    out_df = pd.DataFrame({
        "Id": test_df_ids,
        "GiaNha_ThucTe": y_test.values.round(2),
        "GiaNha_DuDoan": y_pred.round(2),
        "GiaNha_DinhDang": [f"${val:,.2f}" for val in y_pred],
        "ChenhLech_USD": abs_diff.round(2),
        "TiLe_SaiLech_%": [f"{p:.2f}%" for p in pct_diff],
    })

    out_file = output_dir / f"GiaNha_{best_name}.csv"
    out_df.to_csv(out_file, index=False)
    print(f"[✓] Đã lưu file kết quả dự đoán (format chuẩn) vào: {out_file}")


if __name__ == "__main__":
    main()
