# Đề xuất đặc trưng và bảng nhóm biến Ames Housing

**Người phụ trách:** Dương Hoàng Anh — MSSV 0364169.

**Mốc công việc:** 21–23/11/2026.

**Phạm vi:** Đề xuất đặc trưng mới và bảng nhóm biến để bàn giao cho phần xây Pipeline.

Đề xuất ưu tiên 9 đặc trưng về quy mô, tuổi nhà và trạng thái tiện nghi; 12 đặc trưng bổ sung được chia thành nhóm để thử nghiệm sau. Bảng biến gốc bao phủ đủ 79 biến đầu vào, mỗi biến xuất hiện đúng một lần. Việc giữ hoặc bỏ đặc trưng sẽ được quyết định bằng cross-validation trên tập train khi nhóm có Pipeline.

Các bảng đi kèm có thể mở bằng Excel hoặc xem trong notebook:

- [Bảng nhóm 79 biến gốc](Bang_nhom_bien.csv).
- [Bảng 21 đặc trưng đề xuất](Bang_dac_trung_de_xuat.csv), gồm cột nguồn, công thức, đơn vị, mức ưu tiên và cách xử lý ngoại lệ.
- [Thống kê kiểm tra công thức trên train](Kiem_tra_dac_trung_train.csv).
- [Notebook minh họa và kiểm tra](../../03_SourceCode/notebooks/03_Feature_Proposal_HoangAnh.ipynb).

## 1 Dữ liệu và phạm vi phân tích

Nguồn dữ liệu là `train.csv` của Kaggle House Prices. Trong 81 cột, `Id` là mã định danh và `SalePrice` là biến mục tiêu; 79 cột còn lại là biến đầu vào. `SalePrice`, `log1p(SalePrice)` và mọi công thức sử dụng giá bán đều bị loại khỏi phần tạo đặc trưng đầu vào.

Phần khảo sát dùng **1.168 căn thuộc tập train**, xác định bằng cách loại các Id đã lưu trong `data/processed/test_ids.csv`. Giữ nguyên cách chia dữ liệu của nhóm, không chia lại và không đánh giá mô hình trên holdout. Thống kê ở đây được tính trước khi lọc ngoại lai. Nếu áp dụng quy tắc sẵn có `GrLivArea > 4000` của nhóm thì loại 3 căn trên train, còn 1.165 căn.

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
| `HasRemodel` | `1` nếu `YearRemodAdd > YearBuilt`; `0` nếu bằng nhau | Phân biệt nhà từng cải tạo và chưa cải tạo; yêu cầu `YearBuilt <= YearRemodAdd <= YrSold`. |

`TotalSF` bao gồm cả phần tầng hầm chưa hoàn thiện và không cộng riêng `LowQualFinSF`, vì vậy không gọi là diện tích ở đã hoàn thiện hoặc diện tích pháp lý. Không cộng thêm `GrLivArea` vào công thức này vì sẽ đếm trùng phần diện tích trên mặt đất. Các diện tích đều giữ nguyên ft² của dữ liệu để cộng và lấy tỷ số nhất quán.

## 4 Mười hai đặc trưng thử thêm

Thử theo từng nhóm nhỏ để biết nhóm nào tạo thay đổi trong CV. Đặc trưng dư thừa có thể ít hữu ích với mô hình cây nhưng vẫn đáng thử với mô hình tuyến tính.

| Nhóm thử | Đặc trưng | Công thức hoặc quy tắc |
|---|---|---|
| Trạng thái tiện nghi | `HasPool` | `1(PoolArea > 0)` khi biết diện tích; cần chú ý số căn có hồ bơi rất ít. |
| Trạng thái tiện nghi | `Has2ndFloor` | `1(2ndFlrSF > 0)` khi biết diện tích. |
| Tuổi và hoàn thiện | `GarageAge` | `YrSold - GarageYrBlt`, chỉ khi có garage và tuổi không âm. |
| Tuổi và hoàn thiện | `FinishedBsmtSF` | `BsmtFinSF1 + BsmtFinSF2`. |
| Tỷ số | `FinishedBsmtRatio` | `(BsmtFinSF1 + BsmtFinSF2) / TotalBsmtSF`, chỉ khi mẫu số > 0 và kết quả thuộc [0, 1]. |
| Tỷ số | `LivingLotRatio` | `GrLivArea / LotArea`, chỉ khi `LotArea > 0`; có thể lớn hơn 1 với nhà nhiều tầng. |
| Tỷ số | `GarageAreaPerCar` | `GarageArea / GarageCars`, chỉ khi `GarageCars > 0`. |
| Tương tác | `QualArea` | `OverallQual * GrLivArea`, kiểm tra điểm chất lượng trong [1, 10]. |
| Biến đổi log | `LogGrLivArea` | `log1p(GrLivArea)` với diện tích không âm. |
| Biến đổi log | `LogLotArea` | `log1p(LotArea)` với diện tích đất > 0. |
| Chu kỳ tháng | `MoSoldSin` | `sin(2 * pi * (MoSold - 1) / 12)`. |
| Chu kỳ tháng | `MoSoldCos` | `cos(2 * pi * (MoSold - 1) / 12)`; luôn thử cùng biến sin. |

## 5 Quy tắc thiếu dữ liệu và ngoại lệ

1. **Tuổi nhà dùng năm bán của từng dòng.** Không dùng năm hiện tại của máy. Các năm phải là số nguyên dương; tuổi âm được chuyển thành NA và giữ nguyên cột nguồn để đối chiếu. Trên train, căn `Id=524` có `YrSold=2007`, `YearBuilt=2007`, `YearRemodAdd=2008`. Vì vậy `RemodelAge` và `HasRemodel` của căn này được để NA, còn `HouseAge=0`.
2. **Không có hạng mục khác với không biết.** Với chỉ báo có/không, chỉ trả về 0 khi có bằng chứng giá trị nguồn bằng 0. Nếu không đủ bằng chứng về sự hiện diện hoặc vắng mặt thì giữ NA. Với garage, một trong hai giá trị dương đủ để xác định có garage.
3. **Không mặc định mọi thiếu số thành 0.** Công thức cộng giữ NA nếu thiếu một thành phần, tương đương `sum(..., skipna=False)`. Số 0 đã ghi trong nguồn vẫn được giữ. Các giá trị âm của diện tích hoặc số lượng bị coi là không hợp lệ.
4. **Không chia cho 0.** Mẫu số <= 0 hoặc thiếu sẽ cho tỷ số NA. Không có hầm thì `FinishedBsmtRatio=NA` kèm `HasBsmt=0`; không có garage thì `GarageAge` và `GarageAreaPerCar` là NA kèm `HasGarage=0`. Tuổi garage bằng 0 phải dành cho garage mới xây.
5. **Đối chiếu nhãn NA với diện tích hoặc số lượng.** Ví dụ, `FireplaceQu=NA` và `Fireplaces=0` được biểu diễn là `None`. Nếu số lượng > 0 nhưng chất lượng thiếu thì dùng `Unknown` hoặc cách impute đã thống nhất, tránh biến thành không có lò sưởi. Áp dụng tương tự cho garage, tầng hầm và hồ bơi.
6. **Giữ chuỗi `None` thật trong `MasVnrType`.** Dùng hàm `load_raw()` sẵn có để không biến giá trị này thành NaN. Thiếu thật ở `LotFrontage`, `MasVnrArea`, `Electrical` cần impute bằng tham số học trên fold train. Nếu thử median `LotFrontage` theo `Neighborhood`, bảng median và giá trị dự phòng cũng phải học lại trong từng fold.
7. **Tháng bán phải thuộc 1–12 và là số nguyên.** Hai công thức chu kỳ giữ NA khi tháng không hợp lệ. Dùng cả sin và cos để tránh mất thông tin vị trí trong chu kỳ.
8. **Xác định thông tin có sẵn lúc dự đoán.** Trong bài Kaggle, dùng trường năm/tháng bán được cung cấp. Nếu triển khai dự đoán trước giao dịch, cần thống nhất mốc thời gian dự kiến và khả năng có sẵn `SaleType`, `SaleCondition`; không lấy thông tin tương lai thực tế để bổ sung cho đầu vào.

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

Ở nhóm thử thêm, `GarageAge` và `GarageAreaPerCar` có 64 NA tương ứng các căn không có garage; `FinishedBsmtRatio` có 28 NA tương ứng các căn không có tầng hầm. Bảng thống kê đầy đủ 21 đặc trưng nằm trong CSV đi kèm. Các giá trị lớn trong bảng vẫn bao gồm những căn ngoại lai thuộc train; đây không phải ngưỡng tự động để xóa dòng.

## 7 Bàn giao để tích hợp Pipeline

Hoàng Anh bàn giao định nghĩa biến, công thức và minh họa kiểm tra. Hinh tích hợp các quy tắc được chọn vào custom transformer trong `03_SourceCode/src/`, trước `ColumnTransformer`. Hàm `preview_features()` hiện trả về riêng các biến đề xuất để khảo sát, chưa phải một estimator dùng để lưu cùng mô hình.

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
.venv\Scripts\python.exe -m notebook notebooks/03_Feature_Proposal_HoangAnh.ipynb
```

Chọn kernel **Ames BTL (Python 3.11)** khi mở notebook. Lệnh đầu tái tạo ba bảng CSV trong `01_Report/HoangAnh/`; notebook trình bày bảng nhóm, công thức, thống kê và ví dụ tính trên train. Các bảng được ghi UTF-8 có BOM để mở bằng Excel giữ đúng tiếng Việt. Trong Excel có thể dùng Data → From Text/CSV và chọn dấu phẩy làm dấu phân cách.

## 9 Nguồn đối chiếu

- [README của nhóm](../../README.md), phần phân công, lịch trình và quy tắc chống rò rỉ dữ liệu.
- [Hướng dẫn dữ liệu](../../03_SourceCode/data/README.md) và file `03_SourceCode/data/raw/data_description.txt` tải theo hướng dẫn, dùng để diễn giải các cột gốc. Công thức gộp và cách chia nhóm trong phần bàn giao này là đề xuất để nhóm thử nghiệm.
- [Mã đọc dữ liệu và cách chia cố định](../../03_SourceCode/src/data.py).
- [Mã kiểm tra công thức và xuất bảng](../../03_SourceCode/src/feature_proposal.py).
- [scikit-learn Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), quy tắc fit trên train và dùng Pipeline trong CV.
- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data), xử lý các nhóm biến khác kiểu.
