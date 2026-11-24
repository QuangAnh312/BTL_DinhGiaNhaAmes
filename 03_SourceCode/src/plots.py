"""Hàm vẽ biểu đồ đánh giá mô hình 4 góc phục vụ báo cáo và phân tích lỗi.

Dùng cho mô hình cuối sau khi tuning (phần của Quang Anh).
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_model_evaluation(name, y_true, y_pred, metrics, save_path=None):
    """Vẽ sơ đồ đánh giá 4 góc trực quan cho một mô hình hồi quy.

    Parameters
    ----------
    name : str
        Tên mô hình (ví dụ: 'Lasso', 'Ridge').
    y_true : array-like
        Giá trị thực tế của SalePrice (USD).
    y_pred : array-like
        Giá trị dự đoán của SalePrice (USD).
    metrics : dict
        Dictionary chứa các chỉ số: 'r2', 'rmsle', 'rmse', 'mae'.
    save_path : str or Path, optional
        Đường dẫn file ảnh (.png) để lưu. Nếu None thì không lưu.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Đối tượng Figure chứa 4 biểu đồ.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"SƠ ĐỒ ĐÁNH GIÁ MÔ HÌNH: {name.upper()}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="k",
        linewidth=0.5,
        label="Mẫu kiểm tra",
    )
    min_val = min(y_true.min(), y_pred.min()) * 0.9
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Đường chuẩn lý tưởng (y = x)",
    )
    ax1.set_xlabel("Giá thực tế (USD)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Giá dự đoán (USD)", fontsize=11, fontweight="bold")
    title_ax1 = (
        f"Giá thực tế vs Giá dự đoán\n"
        f"R² = {metrics['r2']:.4f} | RMSLE = {metrics['rmsle']:.4f}"
    )
    ax1.set_title(title_ax1, fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    ax1.yaxis.set_major_formatter("${x:,.0f}")

    # 2. Residuals vs Predicted
    ax2 = axes[0, 1]
    ax2.scatter(
        y_pred,
        residuals,
        alpha=0.6,
        color="#2ca02c",
        edgecolor="k",
        linewidth=0.5,
    )
    ax2.axhline(0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Giá dự đoán (USD)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(
        "Phần dư (Y_thực tế - Y_dự đoán) [USD]", fontsize=11, fontweight="bold"
    )
    ax2.set_title("Biểu đồ phần dư (Residuals vs Predicted)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.xaxis.set_major_formatter("${x:,.0f}")
    ax2.yaxis.set_major_formatter("${x:,.0f}")

    # 3. Residual Distribution
    ax3 = axes[1, 0]
    ax3.hist(
        residuals,
        bins=30,
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(min(residuals), max(residuals), 100)
    p = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * std**2))
    ax3.plot(
        x,
        p,
        "k-",
        linewidth=2,
        label=f"Phân phối chuẩn (μ={mu:,.0f}, σ={std:,.0f})",
    )
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
        [
            "Sai số trung vị (Median Absolute Error)",
            f"${np.median(np.abs(residuals)):,.2f}",
        ],
        ["Số mẫu đánh giá", f"{len(y_true):,d} căn"],
    ]
    table = ax4.table(
        cellText=summary_data,
        colWidths=[0.55, 0.45],
        loc="center",
        cellLoc="left",
    )
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
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
