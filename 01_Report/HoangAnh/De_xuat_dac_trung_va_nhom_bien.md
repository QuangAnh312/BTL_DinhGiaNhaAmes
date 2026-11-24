# Đề xuất đặc trưng và bảng nhóm biến Ames Housing

**Người phụ trách:** Dương Hoàng Anh — MSSV 0364169.

**Mốc công việc:** 21–23/11/2026.

**Phạm vi:** Đề xuất đặc trưng mới và bảng nhóm biến để bàn giao cho phần xây Pipeline.

Đề xuất ưu tiên 9 đặc trưng về quy mô, tuổi nhà và trạng thái tiện nghi; 12 đặc trưng bổ sung được chia thành nhóm để thử nghiệm sau. Bảng biến gốc bao phủ đủ 79 biến đầu vào, mỗi biến xuất hiện đúng một lần. Việc giữ hoặc bỏ đặc trưng sẽ được quyết định bằng cross-validation trên tập train khi nhóm có Pipeline.

Các bảng đi kèm có thể mở bằng Excel hoặc xem trong notebook:

- [Bảng nhóm 79 biến gốc](Bang_nhom_bien.csv).
- [Bảng 21 đặc trưng đề xuất](Bang_dac_trung_de_xuat.csv), gồm cột nguồn, công thức, đơn vị, mức ưu tiên và cách xử lý ngoại lệ.
- [Thống kê 9 đặc trưng cốt lõi trên train](Kiem_tra_dac_trung_train.csv).
- [Notebook nháp minh họa](../../03_SourceCode/notebooks/drafts/03_Feature_Proposal_HoangAnh.ipynb).

Code minh họa chỉ tính 9 đặc trưng cốt lõi bằng pandas. Bảng 79 biến và bảng 21 đề xuất được soạn trực tiếp trong các file CSV; 12 ý tưởng thử thêm được giữ để nhóm cân nhắc khi có kết quả CV.

## 1 Dữ liệu và phạm vi phân tích

Nguồn dữ liệu là `train.csv` của Kaggle House Prices. Trong 81 cột, `Id` là mã định danh và `SalePrice` là biến mục tiêu; 79 cột còn lại là biến đầu vào. `SalePrice`, `log1p(SalePrice)` và mọi công thức sử dụng giá bán đều bị loại khỏi phần tạo đặc trưng đầu vào.

Phần khảo sát dùng **1.168 căn thuộc `X_train` do `load_split()` trong `src/data.py` trả về**. Hàm chung này dùng danh sách `data/processed/test_ids.csv` của nhóm. Notebook và lệnh xuất thống kê đều gọi `load_split()`, không chia lại dữ liệu và không đánh giá mô hình trên holdout. Thống kê ở đây được tính trước khi lọc ngoại lai. Nếu áp dụng quy tắc sẵn có `GrLivArea > 4000` của nhóm thì loại 3 căn trên train, còn 1.165 căn.

Theo kiểu lưu trữ, train có **36 cột số và 43 cột nhãn**. Kiểu lưu trữ cần được phân biệt với ý nghĩa: `MSSubClass` là mã phân loại, `MoSold` là tháng có chu kỳ, còn `OverallQual` và `OverallCond` là điểm thứ bậc.

## 2 Bảng nhóm biến gốc

Một biến chỉ thuộc một nhóm chính trong bảng này; một công thức tạo đặc trưng có thể sử dụng biến từ nhiều nhóm. Ví dụ, `GarageYrBlt` thuộc nhóm thời gian, còn `BsmtFullBath` thuộc nhóm tầng hầm.

| Nhóm | Số biến | Các biến gốc |
|---|---:|---|
| Vị trí và lô đất | 12 | `MSZoning`, `LotFrontage`, `LotArea`, `Street`, `Alley`, `LotShape`, `LandContour`, `LotConfig`, `LandSlope`, `Neighborhood`, `Condition1`, `Condition2` |
| Loại hình nhà | 3 | `MSSubClass`, `BldgType`, `HouseStyle` |
| Chất lượng và kết cấu | 11 | `OverallQual`, `OverallCond`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `MasVnrArea`, `ExterQual`, `ExterCond`, `Foundation` |
| Thời gian | 5 | `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`, `MoSold`, `YrSold` |
| Diện tích và phòng trên mặt đất | 8 | `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`, `BedroomAbvGr`, `KitchenAbvGr`, `KitchenQual`, `TotRmsAbvGrd` |
| Tầng hầm | 11 | `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinSF1`, `BsmtFinType2`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath` |
| Phòng tắm trên mặt đất | 2 | `FullBath`, `HalfBath` |
| Hệ thống và tiện nghi trong nhà | 8 | `Utilities`, `Heating`, `HeatingQC`, `CentralAir`, `Electrical`, `Functional`, `Fireplaces`, `FireplaceQu` |
| Garage và lối xe | 7 | `GarageType`, `GarageFinish`, `GarageCars`, `GarageArea`, `GarageQual`, `GarageCond`, `PavedDrive` |
| Không gian ngoài trời | 8 | `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `PoolQC`, `Fence` |
| Hạng mục khác | 2 | `MiscFeature`, `MiscVal` |
| Điều kiện giao dịch | 2 | `SaleType`, `SaleCondition` |
| **Tổng** | **79** | **Không gồm `Id` và `SalePrice`** |

### Cách đưa các nhóm vào tiền xử lý

Baseline dùng one-hot cho các cột nhãn, chuyển `MSSubClass` và `MoSold` sang phân loại. Khi đó có 34 biến số và 45 biến phân loại trước khi one-hot. `OverallQual` và `OverallCond` vẫn giữ dạng điểm số. Thử cặp sin/cos cho tháng ở một cấu hình riêng, thay cho cách one-hot tháng.

Không điền median cho mã `MSSubClass`: nếu thiếu thì dùng `Unknown` sau khi chuyển sang chuỗi, hoặc dùng mode học trong fold. Tháng không nguyên hoặc ngoài 1–12 được coi là thiếu; nhánh one-hot có thể dùng `Unknown`, còn nhánh sin/cos giữ NA để impute các đặc trưng đã tạo trong fold. Điểm `OverallQual` và `OverallCond` phải là số nguyên trong 1–10; số 0 không phải điểm hợp lệ.

Với nhãn có thứ bậc như `ExterQual`, `KitchenQual`, `GarageFinish`, baseline vẫn có thể dùng one-hot. Chỉ chuyển sang ordinal khi cả nhóm đã thống nhất thứ tự và cách biểu diễn trạng thái không có hạng mục. `Fence` kết hợp loại hàng rào và mức riêng tư nên không tự gán một thứ tự tuyến tính. Các cột gần hằng số như `Utilities` chỉ là ứng viên để thử loại trong CV; không tự xóa theo kết quả holdout.

Bảng CSV ghi riêng ý nghĩa, kiểu ngữ nghĩa, hướng mã hóa và xử lý thiếu cho từng biến. Các biến số cần impute trong từng fold; scale khi dùng mô hình nhạy với thang đo như Ridge hoặc KNN. One-hot cần xử lý nhãn mới khi suy luận, chẳng hạn `handle_unknown='ignore'`.

## 3 Chín đặc trưng ưu tiên

Các giả thuyết dưới đây dựa vào ý nghĩa của dữ liệu. Chúng giúp diễn đạt quan hệ để thử nghiệm, chưa thay thế kết quả so sánh bằng CV.

| Đặc trưng | Công thức | Ý nghĩa và lý do thử |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Tổng quy mô tầng hầm và hai tầng chính, đơn vị ft². Giúp mô hình tuyến tính có một biến tổng diện tích. |
| `HouseAge` | `YrSold - YearBuilt` | Tuổi nhà tại năm bán, đơn vị năm; có thể phản ánh hao mòn và thay đổi tiêu chuẩn xây dựng. |
| `RemodelAge` | `YrSold - YearRemodAdd` | Số năm từ lần cải tạo đến năm bán; bổ sung thông tin mà tuổi xây dựng chưa thể hiện. |
| `TotalBath` | `FullBath + BsmtFullBath + 0.5 * (HalfBath + BsmtHalfBath)` | Số phòng tắm quy đổi ở cả trên và dưới đất. Trọng số 0,5 là quy ước đặc trưng. |
| `TotalOutdoorSF` | `WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | Tổng diện tích sàn gỗ và hiên, đơn vị ft²; không gồm hồ bơi. |
| `HasGarage` | `1` nếu `GarageCars > 0` hoặc `GarageArea > 0`; `0` nếu cả hai bằng 0 | Tách sự hiện diện của garage khỏi quy mô garage. |
| `HasBsmt` | `1` nếu `TotalBsmtSF > 0`; `0` nếu bằng 0 | Phân biệt không có tầng hầm với tầng hầm có diện tích khác nhau. |
| `HasFireplace` | `1` nếu `Fireplaces > 0`; `0` nếu bằng 0 | Biểu diễn trạng thái có lò sưởi; tránh chỉ dựa vào nhãn chất lượng bị thiếu khi không có. |
| `HasRemodel` | `1` nếu `YearRemodAdd > YearBuilt`; `0` nếu bằng nhau | Phân biệt nhà từng cải tạo và chưa cải tạo; để NA ở dòng ghi năm cải tạo sau năm bán. |

`TotalSF` bao gồm cả phần tầng hầm chưa hoàn thiện và không cộng riêng `LowQualFinSF`, vì vậy không gọi là diện tích ở đã hoàn thiện hoặc diện tích pháp lý. Không cộng thêm `GrLivArea` vào công thức này vì sẽ đếm trùng phần diện tích trên mặt đất. Các diện tích đều giữ nguyên ft² của dữ liệu để cộng và lấy tỷ số nhất quán.

## 4 Mười hai đặc trưng thử thêm

Các ý tưởng trong bảng này chưa được tính trong code minh họa. Khi có Pipeline, thử từng nhóm nhỏ để biết nhóm nào tạo thay đổi trong CV. Đặc trưng dư thừa có thể ít hữu ích với mô hình cây nhưng vẫn đáng thử với mô hình tuyến tính.

| Nhóm thử | Đặc trưng | Công thức hoặc quy tắc |
|---|---|---|
| Trạng thái tiện nghi | `HasPool` | `1(PoolArea > 0)` khi biết diện tích; cần chú ý số căn có hồ bơi rất ít. |
| Trạng thái tiện nghi | `Has2ndFloor` | `1(2ndFlrSF > 0)` khi biết diện tích. |
| Tuổi và hoàn thiện | `GarageAge` | `YrSold - GarageYrBlt`, chỉ khi có garage và tuổi không âm. |
| Tuổi và hoàn thiện | `FinishedBsmtSF` | `BsmtFinSF1 + BsmtFinSF2`. |
| Tỷ số | `FinishedBsmtRatio` | `(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF`, chỉ khi mẫu số > 0 và kết quả thuộc [0, 1]. |
| Tỷ số | `LivingLotRatio` | `GrLivArea / LotArea`, chỉ khi `LotArea > 0`; có thể lớn hơn 1 với nhà nhiều tầng. |
| Tỷ số | `GarageAreaPerCar` | `GarageArea / GarageCars`, chỉ khi `GarageCars > 0`. |
| Tương tác | `QualArea` | `OverallQual * GrLivArea`, kết hợp điểm chất lượng và quy mô diện tích. |
| Biến đổi log | `LogGrLivArea` | `log1p(GrLivArea)` với diện tích không âm. |
| Biến đổi log | `LogLotArea` | `log1p(LotArea)` với diện tích đất > 0. |
| Chu kỳ tháng | `MoSoldSin` | `sin(2 * pi * (MoSold - 1) / 12)`. |
| Chu kỳ tháng | `MoSoldCos` | `cos(2 * pi * (MoSold - 1) / 12)`; luôn thử cùng biến sin. |

## 5 Quy tắc thiếu dữ liệu và ngoại lệ

1. **Tuổi nhà dùng năm bán của từng dòng.** Trên train, căn `Id=524` có `YrSold=2007`, `YearBuilt=2007`, `YearRemodAdd=2008`. Vì vậy `RemodelAge` và `HasRemodel` của căn này được để NA, còn `HouseAge=0`. Code giữ nguyên các năm nguồn để đối chiếu.
2. **Các cờ có hoặc không dựa vào số lượng và diện tích.** Trong train hiện tại, các cột số dùng để tạo cờ không bị thiếu. So sánh `> 0` rồi dùng `astype(int)` để đổi đúng/sai thành 1/0; riêng `HasRemodel` giữ NA tại trường hợp năm cải tạo bất nhất ở trên. Khi xây Pipeline dùng cho dữ liệu mới, nhóm cần thống nhất cách xử lý các đầu vào thiếu trước khi tạo cờ.
3. **Công thức cộng giữ NA nếu thiếu một thành phần.** Tổng diện tích hiên dùng `sum(axis=1, skipna=False)`. Số 0 đã ghi trong nguồn vẫn được giữ; phần minh họa không học tham số impute.
4. **Các nhãn thiếu được xử lý khi xây Pipeline.** Ví dụ, `FireplaceQu=NA` cùng `Fireplaces=0` biểu thị không có lò sưởi. Hàm `load_split()` sử dụng cách đọc dữ liệu chung để giữ chuỗi `None` thật trong `MasVnrType`. Các tham số impute cho `LotFrontage`, `MasVnrArea`, `Electrical` phải học trên fold train.
5. **Các tỷ số là ý tưởng thử thêm.** Khi nhóm triển khai, mẫu số bằng 0 cần cho NA thay vì vô cực. Không có hầm đi kèm `HasBsmt=0`; không có garage đi kèm `HasGarage=0`. Code 9 đặc trưng hiện tại chưa tính các tỷ số này.

Trong bài Kaggle, dùng trường năm và tháng bán được cung cấp. Nếu triển khai dự đoán trước giao dịch, nhóm cần thống nhất mốc thời gian dự kiến và thông tin có sẵn về điều kiện giao dịch, tránh lấy thông tin tương lai thực tế làm đầu vào.

## 6 Kết quả kiểm tra trên tập train

Các số sau được tính trên 1.168 căn trước khi lọc ngoại lai. Thống kê dùng các giá trị hợp lệ sau khi áp dụng quy tắc ở trên, không impute, không fit mô hình và không sử dụng nhãn giá bán.

| Đặc trưng | Số NA | Nhỏ nhất | Trung vị | Lớn nhất |
|---|---:|---:|---:|---:|
| `TotalSF` | 0 | 334 | 2.494 | 11.752 |
| `HouseAge` | 0 | 0 | 35 | 136 |
| `RemodelAge` | 1 | 0 | 14 | 60 |
| `TotalBath` | 0 | 1 | 2 | 6 |
| `TotalOutdoorSF` | 0 | 0 | 168 | 1.027 |
| `HasGarage` | 0 | 0 | 1 | 1 |
| `HasBsmt` | 0 | 0 | 1 | 1 |
| `HasFireplace` | 0 | 0 | 1 | 1 |
| `HasRemodel` | 1 | 0 | 0 | 1 |

Bảng CSV đi kèm thống kê đúng 9 đặc trưng cốt lõi. Các giá trị lớn trong bảng vẫn bao gồm những căn ngoại lai thuộc train; đây không phải ngưỡng tự động để xóa dòng.

### Cách giải thích code

- `pd.DataFrame(index=X.index)` giữ cùng thứ tự dòng với dữ liệu train.
- Phép cộng gộp diện tích và số phòng tắm; phép trừ năm bán với năm xây hoặc năm cải tạo cho tuổi nhà.
- So sánh `> 0` xác định có tiện nghi; `astype(int)` chuyển kết quả đúng hoặc sai thành 1 hoặc 0.
- `where(...)` giữ NA cho tuổi cải tạo âm và cờ cải tạo của căn có năm bất nhất.
- `load_split()` lấy cùng tập train mà hai notebook chính của nhóm đang dùng.

## 7 Bàn giao để tích hợp Pipeline

Hoàng Anh bàn giao định nghĩa biến, công thức và minh họa 9 đặc trưng cốt lõi. Hinh tích hợp các quy tắc được chọn vào custom transformer trong `03_SourceCode/src/`, trước `ColumnTransformer`. Hàm `preview_features()` trả về riêng 9 cột để khảo sát, chưa phải một estimator dùng để lưu cùng mô hình.

Trình tự dự kiến là đọc dữ liệu đúng quy tắc NA → tách `Id` và mục tiêu → áp dụng chính sách ngoại lai trên train → tạo đặc trưng theo từng dòng → `ColumnTransformer` để impute, encode, scale → mô hình. Với CV, các bộ impute, encode, scale và mọi bước chọn đặc trưng dựa vào dữ liệu đều phải fit trong từng fold train. Đây là mục đích của việc ghép tiền xử lý và mô hình bằng [Pipeline](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Sau bước thêm biến, chọn lại cột số và phân loại để các cột mới thực sự được đưa vào mô hình; tránh dùng danh sách 79 cột cũ khiến `ColumnTransformer` bỏ các biến vừa thêm. [Tài liệu ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data) mô tả cách xử lý các nhóm cột khác nhau.

Kế hoạch so sánh khi có Pipeline:

| Cấu hình | Nội dung | Điều kiện so sánh |
|---|---|---|
| A | Biến gốc, chưa thêm FE | Cùng tiền xử lý, mô hình và các fold CV |
| B | A cộng 9 đặc trưng cốt lõi | Đo thay đổi so với A |
| C | B cộng lần lượt từng nhóm thử thêm | Tách tác động của tỷ số, tương tác, log và tháng chu kỳ |

Quang Anh và Hinh dùng cùng các fold CV trên train để chọn cấu hình, giữ holdout cho lần đánh giá cuối. Khi huấn luyện trên `log1p(SalePrice)`, cần đổi dự đoán về giá bằng `expm1` trước khi báo cáo RMSE/MAE theo USD; R² và RMSLE không có đơn vị. Ghi rõ thang đo khi báo cáo điểm CV.

## 8 Cách xem và tái tạo các bảng

Từ thư mục gốc của repo, chạy trong CMD:

```cmd
cd 03_SourceCode
.venv\Scripts\python.exe -m src.feature_proposal
.venv\Scripts\python.exe -m notebook notebooks/drafts/03_Feature_Proposal_HoangAnh.ipynb
```

Notebook dùng tên kernel chuẩn **`python3`**, hiển thị **Python 3 (ipykernel)**. Chọn interpreter trong `.venv` của dự án; repo hiện quy định cả nhóm dùng **Python 3.11** trong `HUONG_DAN_BAT_DAU.md` và `requirements.txt`. Tên kernel chuẩn không tự thay đổi phiên bản Python. Nếu nhóm đổi phiên bản chung thì cập nhật các hướng dẫn và môi trường cùng nhau.

Lệnh `python -m src.feature_proposal` tái tạo riêng bảng thống kê 9 đặc trưng trong `Kiem_tra_dac_trung_train.csv`. Bảng nhóm biến và bảng đề xuất là tài liệu CSV được soạn trực tiếp. Các bảng ghi UTF-8 có BOM để Excel đọc đúng tiếng Việt; nếu cần, dùng Data → From Text/CSV và chọn dấu phẩy làm dấu phân cách.

Notebook minh họa nằm trong `notebooks/drafts/` để giữ hai notebook chính 01 và 02 gọn theo cấu trúc nhóm. Khi đóng gói nộp, dùng tài liệu và CSV trong `01_Report/HoangAnh/`; các công thức đã thống nhất sẽ được ghép vào hai notebook chính.

## 9 Nguồn đối chiếu

- [README của nhóm](../../README.md), phần phân công, lịch trình và quy tắc chống rò rỉ dữ liệu.
- [Hướng dẫn dữ liệu](../../03_SourceCode/data/README.md) và file `03_SourceCode/data/raw/data_description.txt` tải theo hướng dẫn, dùng để diễn giải các cột gốc. Công thức gộp và cách chia nhóm trong phần bàn giao này là đề xuất để nhóm thử nghiệm.
- [Mã đọc dữ liệu và cách chia cố định](../../03_SourceCode/src/data.py).
- [Mã kiểm tra công thức và xuất bảng](../../03_SourceCode/src/feature_proposal.py).
- [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), quy tắc fit trên train và dùng Pipeline trong CV.
- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data), xử lý các nhóm biến khác kiểu.
