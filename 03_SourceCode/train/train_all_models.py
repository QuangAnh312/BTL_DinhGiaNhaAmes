"""Script huấn luyện toàn bộ từng mô hình, vẽ sơ đồ đánh giá và xuất file kết quả.

Vị trí: 03_SourceCode/train/train_all_models.py
Tạo ra:
  - 03_SourceCode/output/GiaNha_<TênMôHình>.csv (format giá tiền và sai số)
  - 03_SourceCode/output/plots/DanhGia_<TênMôHình>.png (sơ đồ 4 góc đánh giá chuyên sâu)
"""
import sys
import os
from pathlib import Path

# Đặt cấu hình matplotlib không cần giao diện cửa sổ
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import load_split
from src.pipeline import build_pipeline, get_models

OUTPUT_DIR = ROOT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"


def compute_metrics(y_true, y_pred):
    """Tính các chỉ số đánh giá tiêu chuẩn."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # RMSLE
    y_true_log = np.log1p(np.maximum(y_true, 0))
    y_pred_log = np.log1p(np.maximum(y_pred, 0))
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsle": rmsle}


def plot_model_evaluation(name, y_true, y_pred, metrics, save_path):
    """Vẽ sơ đồ đánh giá 4 góc trực quan cho từng mô hình."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"SƠ ĐỒ ĐÁNH GIÁ MÔ HÌNH: {name.upper()}", fontsize=16, fontweight="bold", y=0.98)

    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, color="#1f77b4", edgecolor="k", linewidth=0.5, label="Mẫu kiểm tra")
    min_val = min(y_true.min(), y_pred.min()) * 0.9
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Đường chuẩn lý tưởng (y = x)")
    ax1.set_xlabel("Giá thực tế (USD)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Giá dự đoán (USD)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Giá thực tế vs Giá dự đoán\nR² = {metrics['r2']:.4f} | RMSLE = {metrics['rmsle']:.4f}", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    ax1.yaxis.set_major_formatter("${x:,.0f}")

    # 2. Residuals vs Predicted
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.6, color="#2ca02c", edgecolor="k", linewidth=0.5)
    ax2.axhline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Giá dự đoán (USD)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Phần dư (Y_thực tế - Y_dự đoán) [USD]", fontsize=11, fontweight="bold")
    ax2.set_title("Biểu đồ phần dư (Residuals vs Predicted)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.xaxis.set_major_formatter("${x:,.0f}")
    ax2.yaxis.set_major_formatter("${x:,.0f}")

    # 3. Residual Distribution
    ax3 = axes[1, 0]
    n, bins, patches = ax3.hist(residuals, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.7, density=True)
    # Normal curve fit
    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(min(residuals), max(residuals), 100)
    p = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))
    ax3.plot(x, p, "k-", linewidth=2, label=f"Phân phối chuẩn (μ={mu:,.0f}, σ={std:,.0f})")
    ax3.set_xlabel("Phần dư (USD)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mật độ xác suất", fontsize=11, fontweight="bold")
    ax3.set_title("Phân phối của phần dư (Residuals Distribution)", fontsize=12)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.xaxis.set_major_formatter("${x:,.0f}")

    # 4. Metrics Summary Table Card
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_data = [
        ["Thuật toán / Mô hình", name],
        ["Hệ số xác định (R² Score)", f"{metrics['r2']:.4f}"],
        ["Độ đo hàm mục tiêu (RMSLE)", f"{metrics['rmsle']:.4f}"],
        ["Sai số bình phương trung bình (RMSE)", f"${metrics['rmse']:,.2f}"],
        ["Sai số tuyệt đối trung bình (MAE)", f"${metrics['mae']:,.2f}"],
        ["Sai số lớn nhất (Max Error)", f"${np.max(np.abs(residuals)):,.2f}"],
        ["Sai số trung vị (Median Absolute Error)", f"${np.median(np.abs(residuals)):,.2f}"],
        ["Số mẫu tập kiểm tra (Test Set)", f"{len(y_true):,d} căn"],
    ]
    table = ax4.table(cellText=summary_data, colWidths=[0.55, 0.45], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if col == 0:
            cell.set_text_props(weight="bold", color="#1b365d")
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_text_props(color="#333333")
            cell.set_facecolor("#ffffff")
        if row == 0:
            cell.set_facecolor("#1b365d")
            cell.set_text_props(weight="bold", color="#ffffff")

    ax4.set_title("BẢNG TỔNG KẾT ĐỘ ĐO ĐÁNH GIÁ ĐỊNH LƯỢNG", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    [+] Đã lưu sơ đồ đánh giá: {save_path.name}")


def train_and_export_all():
    """Huấn luyện từng mô hình và xuất file kết quả + sơ đồ."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Đang nạp dữ liệu train/test qua src.data.load_split()...")
    X_train, X_test, y_train, y_test = load_split()

    # Lọc ngoại lai tập train
    keep = X_train["GrLivArea"] <= 4000
    X_train_clean, y_train_clean = X_train[keep], y_train[keep]
    print(f"[*] Tập train sạch: {len(X_train_clean)} dòng | Tập test: {len(X_test)} dòng")

    test_ids = pd.read_csv(ROOT_DIR / "data" / "processed" / "test_ids.csv")["Id"].values
    models = get_models()

    summary_list = []
    print("\n[*] Bắt đầu huấn luyện và đánh giá từng mô hình:")
    for name, model in models.items():
        print(f"\n---> Huấn luyện mô hình: {name}")
        pipe = build_pipeline(model, ordinal=True)
        pipe.fit(X_train_clean, y_train_clean)

        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        summary_list.append({"Model": name, **metrics})

        # 1. Xuất file kết quả dự đoán đã format
        abs_diff = np.abs(y_test.values - y_pred)
        pct_diff = (abs_diff / y_test.values) * 100

        out_df = pd.DataFrame({
            "Id": test_ids,
            "GiaNha_ThucTe": np.round(y_test.values, 2),
            "GiaNha_DuDoan": np.round(y_pred, 2),
            "GiaNha_DinhDang": [f"${val:,.2f}" for val in y_pred],
            "ChenhLech_USD": np.round(abs_diff, 2),
            "TiLe_SaiLech_%": [f"{p:.2f}%" for p in pct_diff],
        })

        out_csv = OUTPUT_DIR / f"GiaNha_{name}.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"    [✓] Đã xuất file dự đoán: {out_csv.name} ({len(out_df)} dòng)")

        # 2. Xuất sơ đồ đánh giá
        plot_path = PLOTS_DIR / f"DanhGia_{name}.png"
        plot_model_evaluation(name, y_test.values, y_pred, metrics, plot_path)

    # Xuất bảng tổng hợp so sánh ra console
    summary_df = pd.DataFrame(summary_list).sort_values("rmsle")
    print("\n" + "=" * 65)
    print("BẢNG XẾP HẠNG 7 MÔ HÌNH TRÊN TẬP KIỂM TRA ĐỘC LẬP (TEST SET):")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    train_and_export_all()
