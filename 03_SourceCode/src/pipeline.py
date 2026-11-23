"""Hàm xây dựng Pipeline: missing -> (ordinal) -> FE -> ColumnTransformer -> mô hình.

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from src.transformers import FeatureAdder, MissingHandler, OrdinalMapper
except ImportError:
    from .transformers import FeatureAdder, MissingHandler, OrdinalMapper

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer chọn cột theo kiểu dữ liệu (tự nhận cột do FE tạo ra)."""
    # Nhánh số: điền median -> chuẩn hóa
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Nhánh phân loại: NA nghĩa là "không có" (PoolQC, Alley, Garage...)
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
    return ColumnTransformer(
        [
            ("num", num_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipe, make_column_selector(dtype_exclude=np.number)),
        ]
    )


def build_pipeline(model, ordinal: bool = True) -> TransformedTargetRegressor:
    """missing -> (ordinal) -> FE -> prep -> model.

    Huấn luyện trên log1p(y), dự đoán trả về USD.
    ordinal=True : cột chất lượng mã hóa thứ bậc (ExterQual, KitchenQual...).
    ordinal=False: các cột đó đi vào one-hot (để so sánh trong báo cáo).
    Khi tuning, tên tham số có tiền tố regressor__model__...
    """
    steps = [("missing", MissingHandler())]
    if ordinal:
        steps.append(("ordinal", OrdinalMapper()))
    steps += [
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
