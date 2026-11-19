# Dữ liệu

**Nguồn:** Kaggle – *House Prices: Advanced Regression Techniques*
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

**Trích dẫn gốc:** De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).

## Cách tải
1. Đăng nhập Kaggle, chấp nhận luật cuộc thi, tải `train.csv` và `data_description.txt`.
2. Đặt vào `data/raw/`:
```
data/raw/train.csv
data/raw/data_description.txt
```
> Chỉ dùng `train.csv` (có cột `SalePrice`). `test.csv` của Kaggle không có nhãn nên không dùng.

> `data/raw/` không commit lên GitHub, nhưng **khi nộp LMS thì chép kèm `train.csv`** (~450 KB) để người chấm chạy lại notebook không cần tải.

## Mô tả nhanh
- 1.460 dòng × 81 cột (`Id`, 79 đặc trưng, `SalePrice`).
- Ý nghĩa từng cột và giá trị NA: xem `data_description.txt`.
