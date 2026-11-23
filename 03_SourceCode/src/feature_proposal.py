"""Chín đặc trưng cốt lõi để Hoàng Anh minh họa và bàn giao cho Pipeline.

Chạy từ 03_SourceCode: python -m src.feature_proposal
Bảng nhóm biến và các ý tưởng thử thêm được soạn trong 01_Report/HoangAnh/.
"""

import pandas as pd

from src.data import ROOT, load_split

REPORT_DIR = ROOT.parent / "01_Report" / "HoangAnh"


def preview_features(X):
    """Tính trực tiếp trên train; không fit, không sửa dữ liệu đầu vào."""
    features = pd.DataFrame(index=X.index)
    features["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    features["HouseAge"] = X["YrSold"] - X["YearBuilt"]
    features["RemodelAge"] = X["YrSold"] - X["YearRemodAdd"]
    # Train có một căn ghi năm cải tạo sau năm bán: giữ NA để đối chiếu.
    features["RemodelAge"] = features["RemodelAge"].where(features["RemodelAge"] >= 0)
    features["TotalBath"] = (
        X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["HalfBath"] + X["BsmtHalfBath"])
    )
    porch_columns = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]
    features["TotalOutdoorSF"] = X[porch_columns].sum(axis=1, skipna=False)
    features["HasGarage"] = ((X["GarageCars"] > 0) | (X["GarageArea"] > 0)).astype(int)
    features["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
    features["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
    features["HasRemodel"] = (X["YearRemodAdd"] > X["YearBuilt"]).astype(int)
    features["HasRemodel"] = features["HasRemodel"].where(
        features["RemodelAge"].notna()
    )
    return features


if __name__ == "__main__":
    X_train, _, _, _ = load_split()
    features = preview_features(X_train)
    summary = pd.DataFrame(
        {
            "so_dong_train": len(X_train),
            "so_thieu": features.isna().sum(),
            "nho_nhat": features.min(),
            "trung_vi": features.median(),
            "lon_nhat": features.max(),
        }
    )
    summary.index.name = "dac_trung"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "Kiem_tra_dac_trung_train.csv", encoding="utf-8-sig")
    print(f"Training rows: {len(X_train)}; core features: {len(features.columns)}")
    print("Updated 01_Report/HoangAnh/Kiem_tra_dac_trung_train.csv")
