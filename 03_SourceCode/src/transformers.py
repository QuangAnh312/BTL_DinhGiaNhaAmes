"""Custom transformer: xử lý missing, mã hóa thứ bậc, Feature Engineering.

Đặt trong src/ (không định nghĩa trong notebook) để predict.py load được .joblib.
- MissingHandler: học (median LotFrontage theo khu, mode) từ train trong fit().
- OrdinalMapper, FeatureAdder: không học gì từ dữ liệu, không gây rò rỉ.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Cột số nhưng bản chất là nhãn -> ép sang chuỗi để đi vào nhánh phân loại
NUM_BUT_CAT = ["MSSubClass", "MoSold", "YrSold"]


# ---- Missing loại A: NA nghĩa là "không có" (theo data_description.txt) ----
# Cột phân loại: NA -> "None"
NONE_CAT_COLS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
# Cột số tương ứng: NA -> 0
ZERO_NUM_COLS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

# ---- Missing loại B: khuyết thật ----
# Điền mode học từ train. Với cột Bsmt*, chỉ điền khi căn nhà CÓ tầng hầm
# (BsmtQual không NaN); nếu không có tầng hầm thì NA -> "None" (loại A).
MODE_COLS = ["Electrical", "BsmtExposure", "BsmtFinType2"]
BSMT_MODE_COLS = ["BsmtExposure", "BsmtFinType2"]

# ---- Mã hóa thứ bậc (None = 0 là mức thấp nhất) ----
_QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    **{
        c: _QUAL
        for c in [
            "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
        ]
    },
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "CentralAir": {"N": 0, "Y": 1},
}


class MissingHandler(BaseEstimator, TransformerMixin):
    """Xử lý missing theo ý nghĩa của cột (không dùng một quy tắc cho tất cả).

    Học từ train (trong fit): median LotFrontage theo Neighborhood, mode.
    Nằm trong Pipeline nên CV fit lại từng fold -> không rò rỉ.
    """

    def fit(self, X, y=None):
        if {"LotFrontage", "Neighborhood"} <= set(X.columns):
            self.lot_by_nbhd_ = (
                X.groupby("Neighborhood")["LotFrontage"].median().dropna().to_dict()
            )
        else:
            self.lot_by_nbhd_ = {}

        if "LotFrontage" in X.columns and pd.notna(X["LotFrontage"].median()):
            self.lot_median_ = float(X["LotFrontage"].median())
        else:
            self.lot_median_ = 0.0

        self.modes_ = {
            c: X[c].mode(dropna=True).iloc[0]
            for c in MODE_COLS
            if c in X.columns and X[c].notna().any()
        }
        return self

    def transform(self, X):
        X = X.copy()

        # Loại B: khuyết thật -> mode (làm trước để phân biệt với "không có")
        for c, mode in self.modes_.items():
            if c not in X.columns:
                continue
            # Đảm bảo kiểu object để gán chuỗi an toàn trên Pandas 3.0+
            if X[c].dtype != "object":
                X[c] = X[c].astype("object")
            mask = X[c].isna()
            if c in BSMT_MODE_COLS and "BsmtQual" in X.columns:
                mask &= X["BsmtQual"].notna()  # chỉ khi có tầng hầm
            X.loc[mask, c] = mode

        # Loại B: LotFrontage -> median theo khu, khu lạ -> median chung
        if "LotFrontage" in X.columns:
            fill = pd.Series(np.nan, index=X.index)
            if "Neighborhood" in X.columns:
                fill = X["Neighborhood"].map(self.lot_by_nbhd_).astype(float)
            fill = fill.fillna(self.lot_median_)
            X["LotFrontage"] = pd.to_numeric(
                X["LotFrontage"], errors="coerce"
            ).fillna(fill)

        # Loại A: NA = "không có"
        for c in NONE_CAT_COLS:
            if c in X.columns:
                if X[c].dtype != "object":
                    X[c] = X[c].astype("object")
                X[c] = X[c].fillna("None")
        for c in ZERO_NUM_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        # Không có garage -> không có năm xây: dùng YearBuilt của nhà
        if {"GarageYrBlt", "YearBuilt"} <= set(X.columns):
            X["GarageYrBlt"] = pd.to_numeric(X["GarageYrBlt"], errors="coerce").fillna(
                pd.to_numeric(X["YearBuilt"], errors="coerce")
            )
        return X


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """Mã hóa các cột chất lượng/thứ bậc thành số nguyên (không học từ dữ liệu).

    Cần chạy SAU MissingHandler (NA đã thành "None"). Giá trị lạ -> 0.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c, mapping in ORDINAL_MAPS.items():
            if c in X.columns:
                X[c] = X[c].map(mapping).fillna(0).astype(float)
        return X


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Tạo đặc trưng mới từ dữ liệu thô của Ames Housing.

    Input : DataFrame thô (đã bỏ Id, SalePrice).
    Output: DataFrame có thêm cột mới; các cột NUM_BUT_CAT đã ép sang str.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def col(name):
            # Cột số thiếu (NaN) nghĩa là "không có" -> 0 khi cộng gộp
            if name not in X.columns:
                return pd.Series(0, index=X.index)
            return pd.to_numeric(X[name], errors="coerce").fillna(0)

        # Tổng diện tích sử dụng
        X["TotalSF"] = col("TotalBsmtSF") + col("1stFlrSF") + col("2ndFlrSF")

        # Tuổi nhà và tuổi từ lần sửa chữa gần nhất (tại thời điểm bán, >= 0)
        X["HouseAge"] = (col("YrSold") - col("YearBuilt")).clip(lower=0)
        X["RemodAge"] = (col("YrSold") - col("YearRemodAdd")).clip(lower=0)

        # Tổng số phòng tắm (phòng tắm nhỏ tính 0.5)
        X["TotalBath"] = (
            col("FullBath")
            + 0.5 * col("HalfBath")
            + col("BsmtFullBath")
            + 0.5 * col("BsmtHalfBath")
        )

        # Tổng diện tích hiên/sân
        X["TotalPorchSF"] = (
            col("OpenPorchSF")
            + col("EnclosedPorch")
            + col("3SsnPorch")
            + col("ScreenPorch")
            + col("WoodDeckSF")
        )

        # Cờ có/không
        X["HasPool"] = (col("PoolArea") > 0).astype(int)
        X["Has2ndFloor"] = (col("2ndFlrSF") > 0).astype(int)
        X["HasGarage"] = (col("GarageArea") > 0).astype(int)
        X["HasFireplace"] = (col("Fireplaces") > 0).astype(int)

        # Ép cột nhãn dạng số sang chuỗi an toàn (không bị dạng "20.0", giữ NaN là NaN)
        for c in NUM_BUT_CAT:
            if c in X.columns:
                X[c] = (
                    pd.to_numeric(X[c], errors="coerce")
                    .apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)
                    .astype("object")
                )
        return X
