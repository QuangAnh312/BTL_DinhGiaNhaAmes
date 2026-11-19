# Mã nguồn – Dự đoán giá nhà Ames Housing

> Sẽ hoàn thiện ở Buổi 6 (người phụ trách: Phạm Tiến Quang).

## Cài đặt
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Tải dữ liệu theo `data/README.md`.

## Chạy
| Việc | Lệnh |
|---|---|
| EDA & tiền xử lý | mở `notebooks/01_EDA_Preprocessing.ipynb` |
| Huấn luyện & đánh giá | mở `notebooks/02_Modeling_Evaluation.ipynb` (xuất `models/best_model.joblib`) |
| Dự đoán dữ liệu mới | `python predict.py --input <file.csv>` |
| Demo web | `streamlit run app/app.py` |
