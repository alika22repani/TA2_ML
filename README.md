# Prediksi Efisiensi BBM Mobil dari Bobot Kendaraan (Regresi Linear Sederhana)

Dataset: Auto MPG (UCI / Kaggle). X = `weight` (lbs), y = `mpg`.

## Menjalankan lokal
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py      # melatih model, menyimpan model/model_mpg.joblib
python app.py        # buka http://127.0.0.1:5000
```

## Deploy (Render)
1. Push repo ke GitHub (folder `model/` dan `static/` ikut di-commit).
2. Render > New > Web Service > pilih repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
