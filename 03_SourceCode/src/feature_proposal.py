"""Bảng biến và công thức đề xuất của Hoàng Anh, chỉ khảo sát trên train.

Chạy từ 03_SourceCode: python -m src.feature_proposal
Các hàm xem trước không fit, không dùng SalePrice, không thay thế Pipeline.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.data import ROOT, TARGET, TEST_IDS_PATH, load_raw

REPORT_DIR = ROOT.parent / "01_Report" / "HoangAnh"

GROUPS = {
    "Vị trí và lô đất": (
        "MSZoning LotFrontage LotArea Street Alley LotShape LandContour "
        "LotConfig LandSlope Neighborhood Condition1 Condition2"
    ).split(),
    "Loại hình nhà": "MSSubClass BldgType HouseStyle".split(),
    "Chất lượng và kết cấu": (
        "OverallQual OverallCond RoofStyle RoofMatl Exterior1st Exterior2nd "
        "MasVnrType MasVnrArea ExterQual ExterCond Foundation"
    ).split(),
    "Thời gian": "YearBuilt YearRemodAdd GarageYrBlt MoSold YrSold".split(),
    "Diện tích và phòng trên mặt đất": (
        "1stFlrSF 2ndFlrSF LowQualFinSF GrLivArea BedroomAbvGr KitchenAbvGr "
        "KitchenQual TotRmsAbvGrd"
    ).split(),
    "Tầng hầm": (
        "BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinSF1 BsmtFinType2 "
        "BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath"
    ).split(),
    "Phòng tắm trên mặt đất": "FullBath HalfBath".split(),
    "Hệ thống và tiện nghi trong nhà": (
        "Utilities Heating HeatingQC CentralAir Electrical Functional "
        "Fireplaces FireplaceQu"
    ).split(),
    "Garage và lối xe": (
        "GarageType GarageFinish GarageCars GarageArea GarageQual "
        "GarageCond PavedDrive"
    ).split(),
    "Không gian ngoài trời": (
        "WoodDeckSF OpenPorchSF EnclosedPorch 3SsnPorch ScreenPorch "
        "PoolArea PoolQC Fence"
    ).split(),
    "Hạng mục khác": "MiscFeature MiscVal".split(),
    "Điều kiện giao dịch": "SaleType SaleCondition".split(),
}

COLUMN_MEANINGS = {
    "MSSubClass": "Mã loại nhà; số là nhãn, không phải đại lượng liên tục",
    "MSZoning": "Phân vùng sử dụng đất",
    "LotFrontage": "Chiều dài mặt tiền tiếp giáp đường, feet",
    "LotArea": "Diện tích lô đất, ft²",
    "Street": "Loại mặt đường",
    "Alley": "Loại đường ngõ; NA theo mô tả là không có ngõ",
    "LotShape": "Hình dạng lô đất",
    "LandContour": "Địa hình lô đất",
    "Utilities": "Các dịch vụ hạ tầng được cung cấp",
    "LotConfig": "Vị trí và cách bố trí lô đất",
    "LandSlope": "Mức độ dốc của đất",
    "Neighborhood": "Khu phố trong Ames",
    "Condition1": "Điều kiện lân cận chính như đường lớn hoặc đường sắt",
    "Condition2": "Điều kiện lân cận bổ sung",
    "BldgType": "Loại công trình nhà ở",
    "HouseStyle": "Kiểu nhà và số tầng",
    "OverallQual": "Điểm chất lượng vật liệu và hoàn thiện, 1 đến 10",
    "OverallCond": "Điểm tình trạng chung, 1 đến 10",
    "YearBuilt": "Năm xây dựng ban đầu",
    "YearRemodAdd": "Năm cải tạo; bằng năm xây nếu không cải tạo",
    "RoofStyle": "Kiểu mái",
    "RoofMatl": "Vật liệu mái",
    "Exterior1st": "Vật liệu bên ngoài chính",
    "Exterior2nd": "Vật liệu bên ngoài bổ sung",
    "MasVnrType": "Loại ốp gạch đá; chuỗi None là không ốp",
    "MasVnrArea": "Diện tích ốp gạch đá, ft²",
    "ExterQual": "Chất lượng vật liệu bên ngoài",
    "ExterCond": "Tình trạng vật liệu bên ngoài",
    "Foundation": "Loại móng",
    "BsmtQual": "Đánh giá chiều cao tầng hầm",
    "BsmtCond": "Tình trạng tầng hầm",
    "BsmtExposure": "Mức độ có mặt thoáng của tầng hầm",
    "BsmtFinType1": "Mức hoàn thiện khu vực tầng hầm thứ nhất",
    "BsmtFinSF1": "Diện tích tầng hầm hoàn thiện loại thứ nhất, ft²",
    "BsmtFinType2": "Mức hoàn thiện khu vực tầng hầm thứ hai",
    "BsmtFinSF2": "Diện tích tầng hầm hoàn thiện loại thứ hai, ft²",
    "BsmtUnfSF": "Diện tích tầng hầm chưa hoàn thiện, ft²",
    "TotalBsmtSF": "Tổng diện tích tầng hầm, ft²",
    "Heating": "Loại hệ thống sưởi",
    "HeatingQC": "Chất lượng và tình trạng hệ thống sưởi",
    "CentralAir": "Có điều hòa trung tâm hay không",
    "Electrical": "Loại hệ thống điện",
    "1stFlrSF": "Diện tích tầng một, ft²",
    "2ndFlrSF": "Diện tích tầng hai, ft²",
    "LowQualFinSF": "Diện tích hoàn thiện chất lượng thấp ở mọi tầng, ft²",
    "GrLivArea": "Diện tích ở trên mặt đất, ft²",
    "BsmtFullBath": "Số phòng tắm đầy đủ dưới tầng hầm",
    "BsmtHalfBath": "Số phòng vệ sinh không đầy đủ dưới tầng hầm",
    "FullBath": "Số phòng tắm đầy đủ trên mặt đất",
    "HalfBath": "Số phòng vệ sinh không đầy đủ trên mặt đất",
    "BedroomAbvGr": "Số phòng ngủ trên mặt đất",
    "KitchenAbvGr": "Số bếp trên mặt đất",
    "KitchenQual": "Chất lượng bếp",
    "TotRmsAbvGrd": "Tổng số phòng trên mặt đất, không tính phòng tắm",
    "Functional": "Mức độ đáp ứng chức năng của nhà",
    "Fireplaces": "Số lò sưởi",
    "FireplaceQu": "Chất lượng lò sưởi; NA có thể là không có lò",
    "GarageType": "Loại và vị trí garage",
    "GarageYrBlt": "Năm xây garage",
    "GarageFinish": "Mức hoàn thiện bên trong garage",
    "GarageCars": "Sức chứa garage theo số xe",
    "GarageArea": "Diện tích garage, ft²",
    "GarageQual": "Chất lượng garage",
    "GarageCond": "Tình trạng garage",
    "PavedDrive": "Mức độ trải mặt lối xe vào",
    "WoodDeckSF": "Diện tích sàn gỗ ngoài trời, ft²",
    "OpenPorchSF": "Diện tích hiên mở, ft²",
    "EnclosedPorch": "Diện tích hiên kín, ft²",
    "3SsnPorch": "Diện tích hiên dùng ba mùa, ft²",
    "ScreenPorch": "Diện tích hiên có lưới, ft²",
    "PoolArea": "Diện tích hồ bơi, ft²",
    "PoolQC": "Chất lượng hồ bơi; NA có thể là không có hồ",
    "Fence": "Loại và chất lượng hàng rào; NA là không có",
    "MiscFeature": "Hạng mục phụ khác; NA là không có theo mô tả",
    "MiscVal": "Giá trị quy đổi của hạng mục phụ, USD; không phải giá bán nhà",
    "MoSold": "Tháng bán, 1 đến 12; có tính chu kỳ",
    "YrSold": "Năm bán; mốc thời gian tính tuổi nhà",
    "SaleType": "Hình thức giao dịch bán",
    "SaleCondition": "Điều kiện giao dịch",
}

ORDERED_LABELS = set(
    (
        "ExterQual ExterCond BsmtQual BsmtCond BsmtExposure BsmtFinType1 "
        "BsmtFinType2 HeatingQC KitchenQual Functional FireplaceQu "
        "GarageFinish GarageQual GarageCond PoolQC LandSlope"
    ).split()
)
YEAR_COLUMNS = {"YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold"}
STRUCTURAL_LABELS = set(
    (
        "BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2 FireplaceQu "
        "GarageType GarageFinish GarageQual GarageCond PoolQC"
    ).split()
)


class FeatureSpec(NamedTuple):
    """Công thức, giả thuyết và quy tắc thiếu cho một biến đề xuất."""

    name: str
    sources: tuple[str, ...]
    formula: str
    unit: str
    priority: str
    meaning: str
    missing: str


PROPOSALS = (
    FeatureSpec(
        "TotalSF",
        ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF"),
        "TotalBsmtSF + 1stFlrSF + 2ndFlrSF",
        "ft²",
        "Cốt lõi",
        "Tổng quy mô tầng hầm và hai tầng chính; gồm hầm chưa hoàn thiện",
        "Thiếu một thành phần thì NA; không cộng thêm GrLivArea",
    ),
    FeatureSpec(
        "HouseAge",
        ("YrSold", "YearBuilt"),
        "YrSold - YearBuilt",
        "năm",
        "Cốt lõi",
        "Tuổi nhà tại năm bán, giả thuyết phản ánh hao mòn",
        "Thiếu năm, năm không hợp lệ hoặc tuổi âm thì NA",
    ),
    FeatureSpec(
        "RemodelAge",
        ("YrSold", "YearRemodAdd"),
        "YrSold - YearRemodAdd",
        "năm",
        "Cốt lõi",
        "Thời gian từ lần cải tạo đến năm bán",
        "Tuổi âm thì NA; giữ nguyên năm gốc để nhóm đối chiếu",
    ),
    FeatureSpec(
        "TotalBath",
        ("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"),
        "FullBath + BsmtFullBath + 0.5 * (HalfBath + BsmtHalfBath)",
        "phòng quy đổi",
        "Cốt lõi",
        "Gộp tiện nghi tắm ở cả trên và dưới đất",
        "Thiếu thành phần thì NA; trọng số 0.5 là quy ước đặc trưng",
    ),
    FeatureSpec(
        "TotalOutdoorSF",
        ("WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"),
        "WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch",
        "ft²",
        "Cốt lõi",
        "Quy mô sàn gỗ và các loại hiên, không gồm hồ bơi",
        "Thiếu thành phần thì NA; số 0 có sẵn được giữ nguyên",
    ),
    FeatureSpec(
        "HasGarage",
        ("GarageCars", "GarageArea"),
        "1 nếu GarageCars > 0 hoặc GarageArea > 0; 0 nếu cả hai = 0",
        "0/1",
        "Cốt lõi",
        "Tách thông tin có garage khỏi quy mô garage",
        "Không đủ bằng chứng có hoặc không có thì NA",
    ),
    FeatureSpec(
        "HasBsmt",
        ("TotalBsmtSF",),
        "1 nếu TotalBsmtSF > 0; 0 nếu = 0",
        "0/1",
        "Cốt lõi",
        "Phân biệt không có hầm với hầm có diện tích khác nhau",
        "Diện tích thiếu hoặc âm thì NA, không tự gán 0",
    ),
    FeatureSpec(
        "HasFireplace",
        ("Fireplaces",),
        "1 nếu Fireplaces > 0; 0 nếu = 0",
        "0/1",
        "Cốt lõi",
        "Biểu diễn tiện nghi lò sưởi theo trạng thái có hay không",
        "Số lượng thiếu hoặc âm thì NA",
    ),
    FeatureSpec(
        "HasRemodel",
        ("YearBuilt", "YearRemodAdd", "YrSold"),
        "1 nếu YearRemodAdd > YearBuilt; 0 nếu bằng nhau",
        "0/1",
        "Cốt lõi",
        "Phân biệt nhà đã cải tạo với nhà chưa cải tạo",
        "Thiếu năm hoặc không thỏa YearBuilt <= YearRemodAdd <= YrSold thì NA",
    ),
    FeatureSpec(
        "HasPool",
        ("PoolArea",),
        "1 nếu PoolArea > 0; 0 nếu = 0",
        "0/1",
        "Thử thêm",
        "Tách tiện nghi hồ bơi; cần kiểm tra độ hiếm trong CV",
        "Diện tích thiếu hoặc âm thì NA",
    ),
    FeatureSpec(
        "Has2ndFloor",
        ("2ndFlrSF",),
        "1 nếu 2ndFlrSF > 0; 0 nếu = 0",
        "0/1",
        "Thử thêm",
        "Phân biệt nhà có tầng hai",
        "Diện tích thiếu hoặc âm thì NA",
    ),
    FeatureSpec(
        "GarageAge",
        ("YrSold", "GarageYrBlt", "GarageCars", "GarageArea"),
        "YrSold - GarageYrBlt khi HasGarage = 1",
        "năm",
        "Thử thêm",
        "Tuổi garage khi bán, có thể khác tuổi nhà",
        "Không garage, thiếu năm hoặc tuổi âm thì NA; giữ HasGarage đi kèm",
    ),
    FeatureSpec(
        "FinishedBsmtSF",
        ("BsmtFinSF1", "BsmtFinSF2"),
        "BsmtFinSF1 + BsmtFinSF2",
        "ft²",
        "Thử thêm",
        "Quy mô phần tầng hầm đã hoàn thiện",
        "Thiếu thành phần thì NA; không có hầm và số liệu đều 0 thì giữ 0",
    ),
    FeatureSpec(
        "FinishedBsmtRatio",
        ("BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF"),
        "(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF",
        "tỷ lệ 0 đến 1",
        "Thử thêm",
        "Tỷ lệ hoàn thiện tầng hầm, bổ sung thông tin ngoài tổng diện tích",
        "Mẫu số <= 0, thiếu dữ liệu hoặc tỷ lệ > 1 thì NA; giữ HasBsmt",
    ),
    FeatureSpec(
        "LivingLotRatio",
        ("GrLivArea", "LotArea"),
        "GrLivArea / LotArea",
        "tỷ số",
        "Thử thêm",
        "Diện tích ở trên mặt đất tương đối với diện tích đất",
        "LotArea <= 0 hoặc thiếu thì NA; tỷ số > 1 vẫn có thể hợp lệ",
    ),
    FeatureSpec(
        "GarageAreaPerCar",
        ("GarageArea", "GarageCars"),
        "GarageArea / GarageCars",
        "ft²/xe",
        "Thử thêm",
        "Độ rộng garage trên mỗi chỗ xe",
        "GarageCars <= 0 hoặc thiếu thì NA; giữ HasGarage đi kèm",
    ),
    FeatureSpec(
        "QualArea",
        ("OverallQual", "GrLivArea"),
        "OverallQual * GrLivArea",
        "điểm × ft²",
        "Thử thêm",
        "Tương tác chất lượng và quy mô, hữu ích để thử với Ridge",
        "Thiếu dữ liệu, chất lượng ngoài 1 đến 10 hoặc diện tích âm thì NA",
    ),
    FeatureSpec(
        "LogGrLivArea",
        ("GrLivArea",),
        "log1p(GrLivArea)",
        "log tự nhiên",
        "Thử thêm",
        "Thử quan hệ diện tích với giá ở thang log",
        "Thiếu hoặc diện tích âm thì NA; không cần học tham số từ dữ liệu",
    ),
    FeatureSpec(
        "LogLotArea",
        ("LotArea",),
        "log1p(LotArea)",
        "log tự nhiên",
        "Thử thêm",
        "Giảm tác động của đuôi dài diện tích đất",
        "Thiếu hoặc LotArea <= 0 thì NA",
    ),
    FeatureSpec(
        "MoSoldSin",
        ("MoSold",),
        "sin(2 * pi * (MoSold - 1) / 12)",
        "-1 đến 1",
        "Thử thêm",
        "Mã hóa mùa vụ; luôn thử cùng MoSoldCos",
        "Tháng không nguyên hoặc ngoài 1 đến 12 thì NA",
    ),
    FeatureSpec(
        "MoSoldCos",
        ("MoSold",),
        "cos(2 * pi * (MoSold - 1) / 12)",
        "-1 đến 1",
        "Thử thêm",
        "Giữ tháng 12 và tháng 1 gần nhau theo chu kỳ",
        "Tháng không nguyên hoặc ngoài 1 đến 12 thì NA",
    ),
)


def read_training_features():
    """Đọc đúng phần train theo Id đã lưu; loại Id và nhãn trước khi khảo sát."""
    raw = load_raw().drop(columns=[TARGET])
    test_ids = pd.read_csv(TEST_IDS_PATH)["Id"]
    return raw.loc[~raw["Id"].isin(test_ids)].drop(columns="Id").copy()


def _column_plan(column, is_numeric):
    """Gợi ý ngữ nghĩa và tiền xử lý; việc fit thuộc Pipeline của nhóm."""
    if column == "MSSubClass":
        return "Mã phân loại", "Đổi sang chuỗi, one-hot; không scale mã số"
    if column == "MoSold":
        return "Tháng có chu kỳ", "Baseline one-hot; thử cặp sin/cos theo CV"
    if column in YEAR_COLUMNS:
        return "Năm", "Tính tuổi theo YrSold; median trong fold khi thiếu"
    if column in {"OverallQual", "OverallCond"}:
        return "Điểm thứ bậc 1 đến 10", "Giữ điểm số; impute và scale trong fold"
    if column in ORDERED_LABELS:
        return "Nhãn có thứ bậc", "Baseline one-hot; ordinal khi chốt thứ tự"
    if is_numeric:
        return "Đại lượng hoặc số đếm", "Median trong fold; scale cho Ridge/KNN"
    return "Phân loại", "One-hot, handle_unknown='ignore'; fit trong fold"


def _missing_plan(column, is_numeric):
    if column == "MSSubClass":
        return (
            "Thiếu thành Unknown sau đổi chuỗi; hoặc mode trong fold, không median mã"
        )
    if column == "MoSold":
        return (
            "Chỉ tháng nguyên 1–12; sai thành NA; one-hot dùng Unknown, sin/cos giữ NA"
        )
    if column == "LotFrontage":
        return "Median trong fold; median theo Neighborhood cũng phải fit trong fold"
    if column == "MasVnrType":
        return "Giữ chuỗi None; NA thật thành Unknown hoặc mode học trong fold"
    if column == "MasVnrArea":
        return "NA và MasVnrType=None có thể điền 0; còn lại median trong fold"
    if column == "GarageYrBlt":
        return "Năm nguyên dương; không garage thì NA; không gán 0; giữ HasGarage"
    if column in YEAR_COLUMNS:
        return "Năm nguyên dương; sai thành NA; tính tuổi trước impute trong fold"
    if column in {"OverallQual", "OverallCond"}:
        return "Điểm nguyên 1–10; sai thành NA; impute trong fold, không mặc định 0"
    if column in STRUCTURAL_LABELS:
        return "NA thành None khi xác định không có hạng mục; còn lại Unknown"
    if column in {"Alley", "Fence"}:
        return "NA thành None theo mô tả dữ liệu; giữ trạng thái không có"
    if column == "MiscFeature":
        return (
            "NA thành None theo mô tả; đối chiếu MiscVal, không suy có/không chỉ từ giá"
        )
    if is_numeric:
        return "Giữ 0 đã ghi; thiếu thật dùng median trong fold, không mặc định 0"
    return "Unknown hoặc most_frequent học trong fold; không xóa dòng"


def feature_group_table(X_train):
    """Mỗi biến gốc xuất hiện đúng một lần, theo thứ tự cột trong CSV."""
    columns = [column for group in GROUPS.values() for column in group]
    if len(columns) != 79 or len(set(columns)) != 79:
        raise ValueError("Variable groups must contain 79 distinct predictors")
    if not X_train.columns.is_unique or set(X_train.columns) != set(columns):
        raise ValueError("Expected the 79 original predictors, without Id or SalePrice")
    lookup = {column: name for name, group in GROUPS.items() for column in group}
    rows = []
    for column in X_train:
        numeric = pd.api.types.is_numeric_dtype(X_train[column])
        semantic, preprocessing = _column_plan(column, numeric)
        rows.append(
            {
                "bien": column,
                "nhom": lookup[column],
                "y_nghia": COLUMN_MEANINGS[column],
                "kieu_trong_csv": "Số" if numeric else "Nhãn",
                "kieu_ngu_nghia": semantic,
                "tien_xu_ly_de_xuat": preprocessing,
                "xu_ly_thieu": _missing_plan(column, numeric),
                "so_thieu_train": int(X_train[column].isna().sum()),
                "ty_le_thieu_train_pct": round(X_train[column].isna().mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def feature_proposal_table():
    """Bảng công thức để nhóm triển khai custom transformer trong src/."""
    return pd.DataFrame(
        [
            {
                "dac_trung": spec.name,
                "bien_nguon": ", ".join(spec.sources),
                "cong_thuc": spec.formula,
                "don_vi": spec.unit,
                "uu_tien": spec.priority,
                "y_nghia_va_gia_thuyet": spec.meaning,
                "xu_ly_thieu_va_ngoai_le": spec.missing,
            }
            for spec in PROPOSALS
        ]
    )


def _numbers(X, columns):
    values = X.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").astype(float)
    return values.where(np.isfinite(values) & (values >= 0))


def _positive_indicator(values):
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[(values > 0).any(axis=1)] = 1.0
    result.loc[(values == 0).all(axis=1)] = 0.0
    return result


def preview_features(X):
    """Tính theo từng dòng, giữ NA, không học thống kê và không sửa X đầu vào.

    Chỉ dùng để kiểm tra đề xuất. Khi huấn luyện, nhóm đưa quy tắc này vào
    custom transformer trước ColumnTransformer trong Pipeline.
    """
    required = {column for spec in PROPOSALS for column in spec.sources}
    if missing := required.difference(X.columns):
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    numbers = _numbers(X, sorted(required))
    years = numbers.loc[:, sorted(YEAR_COLUMNS)]
    years = years.where((years > 0) & (years % 1 == 0))
    result = pd.DataFrame(index=X.index)

    def total(columns):
        # skipna=False tránh cộng thiếu thành phần mà vẫn trả ra tổng nhỏ hơn thật.
        return numbers.loc[:, list(columns)].sum(axis=1, skipna=False)

    def age(later, earlier):
        value = years[later] - years[earlier]
        return value.where(value >= 0)

    result["TotalSF"] = total(("TotalBsmtSF", "1stFlrSF", "2ndFlrSF"))
    result["HouseAge"] = age("YrSold", "YearBuilt")
    result["RemodelAge"] = age("YrSold", "YearRemodAdd")
    result["TotalBath"] = (
        numbers["FullBath"]
        + numbers["BsmtFullBath"]
        + 0.5 * (numbers["HalfBath"] + numbers["BsmtHalfBath"])
    )
    result["TotalOutdoorSF"] = total(
        ("WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch")
    )
    result["HasGarage"] = _positive_indicator(numbers[["GarageCars", "GarageArea"]])
    for name, source in (
        ("HasBsmt", "TotalBsmtSF"),
        ("HasFireplace", "Fireplaces"),
        ("HasPool", "PoolArea"),
        ("Has2ndFloor", "2ndFlrSF"),
    ):
        result[name] = _positive_indicator(numbers[[source]])
    remodel_gap = years["YearRemodAdd"] - years["YearBuilt"]
    valid_remodel = (remodel_gap >= 0) & result["RemodelAge"].notna()
    result["HasRemodel"] = (remodel_gap > 0).astype(float).where(valid_remodel)
    result["GarageAge"] = age("YrSold", "GarageYrBlt").where(result["HasGarage"] == 1)
    result["FinishedBsmtSF"] = total(("BsmtFinSF1", "BsmtFinSF2"))
    basement_area = numbers["TotalBsmtSF"].where(numbers["TotalBsmtSF"] > 0)
    ratio = result["FinishedBsmtSF"] / basement_area
    result["FinishedBsmtRatio"] = ratio.where(ratio <= 1)
    result["LivingLotRatio"] = numbers["GrLivArea"] / numbers["LotArea"].where(
        numbers["LotArea"] > 0
    )
    result["GarageAreaPerCar"] = numbers["GarageArea"] / numbers["GarageCars"].where(
        numbers["GarageCars"] > 0
    )
    quality = numbers["OverallQual"]
    quality = quality.where(quality.between(1, 10) & (quality % 1 == 0))
    result["QualArea"] = quality * numbers["GrLivArea"]
    result["LogGrLivArea"] = np.log1p(numbers["GrLivArea"])
    result["LogLotArea"] = np.log1p(numbers["LotArea"].where(numbers["LotArea"] > 0))
    month = numbers["MoSold"]
    month = month.where(month.between(1, 12) & (month % 1 == 0))
    angle = 2 * np.pi * (month - 1) / 12
    result["MoSoldSin"] = np.sin(angle)
    result["MoSoldCos"] = np.cos(angle)
    return result.loc[:, [spec.name for spec in PROPOSALS]].replace(
        [np.inf, -np.inf], np.nan
    )


def feature_summary(X_train):
    """Thống kê kiểm tra công thức trên train, không dùng nhãn hay holdout."""
    preview = preview_features(X_train)
    return pd.DataFrame(
        {
            "dac_trung": preview.columns,
            "so_dong_train": len(preview),
            "so_thieu": preview.isna().sum().to_numpy(),
            "nho_nhat": preview.min().to_numpy(),
            "trung_vi": preview.median().to_numpy(),
            "lon_nhat": preview.max().to_numpy(),
        }
    )


def export_tables():
    """Xuất các bảng bàn giao, không chép dữ liệu gốc hay tạo mô hình."""
    X_train = read_training_features()
    tables = {
        "Bang_nhom_bien.csv": feature_group_table(X_train),
        "Bang_dac_trung_de_xuat.csv": feature_proposal_table(),
        "Kiem_tra_dac_trung_train.csv": feature_summary(X_train),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / name, index=False, encoding="utf-8-sig")
        print(f"Exported {name}: {len(table)} rows")
    print(f"Training predictors: {X_train.shape}; no target used")


if __name__ == "__main__":
    export_tables()
