"""(a) So ngưỡng ngoại lai công bằng; (b) gộp các phương án được chọn.

Chạy giống so_sanh_phuong_an.py (trên commit 6d32e2e). Chỉ dùng tập train. (a): bỏ ngoại lai CHỈ ở phần huấn luyện của mỗi fold,
phần kiểm tra của fold giữ nguyên -> các quy tắc được chấm trên cùng tập điểm.
"""
import sys
import warnings

from pathlib import Path

import numpy as np
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from so_sanh_phuong_an import MODELS, build  # noqa: E402
from src.data import load_split, remove_outliers  # noqa: E402

X_all, _, y_all, _ = load_split()  # 1168 căn train, CHƯA bỏ ngoại lai
cv = KFold(5, shuffle=True, random_state=42)
rules = {
    "Không bỏ": lambda X, y: np.ones(len(X), bool),
    "GrLivArea > 4000 (3 căn)": lambda X, y: (X["GrLivArea"] <= 4000).to_numpy(),
    "> 4000 và giá < 300k (2 căn)": lambda X, y: (
        ~((X["GrLivArea"] > 4000) & (y < 300000))).to_numpy(),
}
print("(a) Ngưỡng ngoại lai – RMSLE CV, đánh giá trên đủ 1.168 căn")
for rname, rule in rules.items():
    out = {}
    for mname, mk in MODELS.items():
        sc = []
        for tr, va in cv.split(X_all):
            Xt, yt = X_all.iloc[tr], y_all.iloc[tr]
            keep = rule(Xt, yt)
            m = build(mk()).fit(Xt[keep], yt[keep])
            sc.append(root_mean_squared_log_error(y_all.iloc[va], m.predict(X_all.iloc[va])))
        out[mname] = round(float(np.mean(sc)), 4)
    print(f"  {rname:<32}", out)

X, y = remove_outliers(X_all, y_all)
cv5 = KFold(5, shuffle=True, random_state=42)
combos = {
    "Baseline": {},
    "Chốt: log1p + BsmtFin one-hot + bỏ PoolQC": {
        "log": True, "bsmt_fin": False, "pool": False, "drop": ("PoolQC",)},
    "Chốt + MoSold/YrSold dạng số": {
        "log": True, "bsmt_fin": False, "pool": False, "drop": ("PoolQC",),
        "month_num": True},
}
print("\n(b) Gộp phương án – RMSLE CV (1.165 căn, đã bỏ ngoại lai)")
for cname, kw in combos.items():
    out = {}
    for mname, mk in MODELS.items():
        s = -cross_val_score(build(mk(), **kw), X, y, cv=cv5,
                             scoring="neg_root_mean_squared_log_error")
        out[mname] = f"{s.mean():.4f} ± {s.std():.4f}"
    print(f"  {cname:<45}", out)
