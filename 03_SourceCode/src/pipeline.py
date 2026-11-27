"""Hàm xây dựng Pipeline: missing -> ordinal -> FE -> ColumnTransformer -> mô hình.

Mọi bước học từ dữ liệu (impute, scale, one-hot) nằm trong Pipeline,
nên cross-validation fit lại từng fold và không bị rò rỉ.
"""

import numpy as np
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from src.transformers import FeatureAdder, MissingHandler, OrdinalMapper
except ImportError:
    from .transformers import FeatureAdder, MissingHandler, OrdinalMapper

RANDOM_STATE = 42

# 8 biến số lệch phải mạnh (EDA Mục 6.4) -> đưa qua nhánh log1p riêng
SKEWED_COLS = [
    "LotArea",
    "GrLivArea",
    "1stFlrSF",
    "LotFrontage",
    "OpenPorchSF",
    "MasVnrArea",
    "WoodDeckSF",
    "2ndFlrSF",
]


def _select_num_cols(df):
    """Chọn các cột số thông thường (loại trừ SKEWED_COLS để tránh trùng lặp)."""
    cols = df.select_dtypes("number").columns
    return [c for c in cols if c not in SKEWED_COLS]


def _select_log_cols(df):
    """Chọn các cột lệch phải trong SKEWED_COLS có mặt trong DataFrame."""
    return [c for c in SKEWED_COLS if c in df.columns]


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer gồm 3 nhánh: num, cat, log (tự nhận cột do FE tạo ra)."""
    # Nhánh số: điền median -> chuẩn hóa
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Nhánh phân loại: NA nghĩa là "không có" (Alley, Garage, BsmtFinType...)
    # -> điền "None" rồi one-hot. dense để HistGradientBoosting dùng được.
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.01,  # gộp hạng hiếm (<1%)
                    sparse_output=False,
                ),
            ),
        ]
    )
    # Nhánh log1p cho 8 biến lệch phải: điền median -> log1p -> chuẩn hóa
    log_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, _select_num_cols),
            ("cat", cat_pipe, make_column_selector(dtype_exclude=np.number)),
            ("log", log_pipe, _select_log_cols),
        ]
    )


def build_pipeline(model, ordinal: bool = True) -> TransformedTargetRegressor:
    """missing -> ordinal -> FE -> prep -> model.

    Huấn luyện trên log1p(y), dự đoán trả về USD.
    ordinal=True : cột chất lượng mã hóa thứ bậc (ExterQual, KitchenQual...).
    ordinal=False: dùng "passthrough" để các cột đó đi vào one-hot.
    Khi tuning, thử regressor__ordinal và regressor__model__...
    """
    steps = [
        ("missing", MissingHandler()),
        ("ordinal", OrdinalMapper() if ordinal else "passthrough"),
        ("fe", FeatureAdder()),
        ("prep", build_preprocessor()),
        ("model", model),
    ]
    return TransformedTargetRegressor(
        regressor=Pipeline(steps), func=np.log1p, inverse_func=np.expm1
    )


def get_models() -> dict:
    """Các mô hình thử nghiệm: baseline -> tuyến tính -> cây -> ensemble."""
    rs = RANDOM_STATE
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=10),
        "Lasso": Lasso(alpha=0.001, max_iter=20000),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=rs),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=rs
        ),
        "HistGB": HistGradientBoostingRegressor(random_state=rs),
    }
