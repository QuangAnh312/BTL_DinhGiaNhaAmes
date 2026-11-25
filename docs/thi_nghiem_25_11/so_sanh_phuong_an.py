"""Thí nghiệm chốt chiến lược tiền xử lý (họp 25/11): mỗi phương án đổi 1 điểm.

Chỉ dùng tập train (đã bỏ ngoại lai), 5-Fold CV cố định. Không dùng tập test.
Baseline là Pipeline TRƯỚC khi chốt (commit 6d32e2e), nên phải chạy trên commit đó:
    git worktree add ../ames_6d32e2e 6d32e2e
    cd ../ames_6d32e2e/03_SourceCode   (chép data/raw/train.csv vào)
    python <đường dẫn tới file này>
Các class ở đây chỉ phục vụ thí nghiệm (không lưu joblib), không dùng cho mô hình nộp.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src import transformers as T  # noqa: E402
from src.data import load_split, remove_outliers  # noqa: E402

X, _, y, _ = load_split()
X, y = remove_outliers(X, y)
CV = KFold(5, shuffle=True, random_state=42)

SKEWED = ["LotArea", "GrLivArea", "1stFlrSF", "LotFrontage", "OpenPorchSF",
          "MasVnrArea", "WoodDeckSF", "2ndFlrSF"]


class Ordinal(BaseEstimator, TransformerMixin):
    def __init__(self, bsmt_fin=True, pool=True):
        self.bsmt_fin = bsmt_fin
        self.pool = pool

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        maps = dict(T.ORDINAL_MAPS)
        if not self.bsmt_fin:
            maps.pop("BsmtFinType1"); maps.pop("BsmtFinType2")
        if not self.pool:
            maps.pop("PoolQC")
        X = X.copy()
        for c, m in maps.items():
            if c in X.columns:
                X[c] = X[c].map(m).fillna(0).astype(float)
        return X


class Extra(BaseEstimator, TransformerMixin):
    """Sau FeatureAdder: đổi MoSold/YrSold về số, thêm cờ, bỏ cột."""

    def __init__(self, month_num=False, flags=False, drop=()):
        self.month_num = month_num
        self.flags = flags
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.month_num:
            for c in ("MoSold", "YrSold"):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        if self.flags:
            X["HasBsmt"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
            X["HasRemodel"] = (X["YearRemodAdd"] > X["YearBuilt"]).astype(int)
        return X.drop(columns=[c for c in self.drop if c in X.columns])


def preprocessor(log=False, min_freq=0.01):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="None")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq,
                             sparse_output=False)),
    ])
    logp = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())])

    def num_cols(df):
        cols = df.select_dtypes("number").columns
        return [c for c in cols if not (log and c in SKEWED)]

    def cat_cols(df):
        return list(df.select_dtypes(exclude="number").columns)

    parts = [("num", num, num_cols), ("cat", cat, cat_cols)]
    if log:
        parts.append(("log", logp, lambda df: [c for c in SKEWED if c in df.columns]))
    return ColumnTransformer(parts)


def build(model, bsmt_fin=True, pool=True, month_num=False, flags=False, drop=(),
          log=False, min_freq=0.01):
    steps = [("missing", T.MissingHandler()), ("ordinal", Ordinal(bsmt_fin, pool)),
             ("fe", T.FeatureAdder()), ("extra", Extra(month_num, flags, drop)),
             ("prep", preprocessor(log, min_freq)), ("model", model)]
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log1p,
                                      inverse_func=np.expm1)


MODELS = {
    "Ridge": lambda: Ridge(alpha=10),
    "Lasso": lambda: Lasso(alpha=0.001, max_iter=20000),
    "HistGB": lambda: HistGradientBoostingRegressor(random_state=42),
}
VARIANTS = {
    "0. Baseline (Pipeline hiện tại)": {},
    "1. BsmtFinType1/2 -> one-hot": {"bsmt_fin": False},
    "2. log1p cho 8 biến lệch": {"log": True},
    "3. Bỏ Utilities, Street": {"drop": ("Utilities", "Street")},
    "4. Bỏ PoolQC (giữ HasPool)": {"pool": False, "drop": ("PoolQC",)},
    "5. MoSold, YrSold giữ dạng số": {"month_num": True},
    "6. Bỏ GarageYrBlt": {"drop": ("GarageYrBlt",)},
    "7. min_frequency = 10 căn": {"min_freq": 10},
    "8. Không gộp mức hiếm": {"min_freq": None},
    "9. Thêm HasBsmt, HasRemodel": {"flags": True},
}

def main():
    rows = []
    base = {}
    for vname, kw in VARIANTS.items():
        row = {"Phuong_an": vname}
        for mname, mk in MODELS.items():
            s = -cross_val_score(build(mk(), **kw), X, y, cv=CV,
                                 scoring="neg_root_mean_squared_log_error")
            if vname.startswith("0."):
                base[mname] = s
            row[mname] = s.mean()
            row[mname + "_fold_tot_hon"] = int((s < base[mname] - 1e-9).sum())
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows).set_index("Phuong_an")
    for m in MODELS:
        df[m + "_chenh"] = df[m] - df.loc[df.index[0], m]
    pd.set_option("display.width", 200)
    print(df[[c for m in MODELS for c in (m, m + "_chenh", m + "_fold_tot_hon")]].round(4))
    print("Do lech chuan RMSLE giua fold (baseline):",
          {m: round(base[m].std(), 4) for m in MODELS})
    if len(sys.argv) > 1:
        df.to_csv(sys.argv[1], encoding="utf-8-sig")


if __name__ == "__main__":
    main()
