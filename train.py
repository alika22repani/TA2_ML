"""Training model Regresi Linear Sederhana: weight (X) -> mpg (y).
Jalankan:  python train.py
Output  :  model/model_mpg.joblib, model/metrics.json, static/scatter.png, static/actual_vs_pred.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

DATA_PATH = "data/auto-mpg.csv"
FEATURE, TARGET = "weight", "mpg"

# 1. Load data ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH, na_values=["?"])
df.columns = [c.strip().lower() for c in df.columns]
print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- Nama Kolom ---")
print(df.columns)
print("\n--- Info & Missing Values ---")
print(df[[FEATURE, TARGET]].describe())
print(df[[FEATURE, TARGET]].isna().sum())

# 2. Preprocessing -----------------------------------------------------------
data = df[[FEATURE, TARGET]].dropna().copy()
print(f"\nJumlah data setelah dropna: {len(data)}")

# Deteksi outlier dengan IQR (dilaporkan; data tidak dibuang karena masih wajar)
q1, q3 = data[TARGET].quantile([0.25, 0.75])
iqr = q3 - q1
n_out = ((data[TARGET] < q1 - 1.5 * iqr) | (data[TARGET] > q3 + 1.5 * iqr)).sum()
print(f"Outlier mpg (IQR): {n_out}")

corr = data[FEATURE].corr(data[TARGET])
print(f"Korelasi Pearson weight vs mpg: {corr:.4f}")

# 3. EDA: scatterplot validasi asumsi linearitas -----------------------------
plt.figure(figsize=(7, 5))
sns.regplot(data=data, x=FEATURE, y=TARGET, scatter_kws={"alpha": .5}, line_kws={"color": "red"})
plt.title(f"Bobot Kendaraan vs Efisiensi BBM (r = {corr:.2f})")
plt.xlabel("Bobot kendaraan (lbs)")
plt.ylabel("Efisiensi bahan bakar (mpg)")
plt.tight_layout()
plt.savefig("static/scatter.png", dpi=120)
plt.close()

# 4. Split, train, evaluasi --------------------------------------------------
X = data[[FEATURE]]
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print(f"\nMAE : {mae:.4f}\nRMSE: {rmse:.4f}\nR2  : {r2:.4f}")
print(f"Intercept (b0): {model.intercept_:.4f}")
print(f"Koefisien (b1): {model.coef_[0]:.6f}  -> per 1000 lbs: {model.coef_[0]*1000:.2f} mpg")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred, alpha=.6)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, color="red", lw=2)
plt.xlabel("mpg aktual"); plt.ylabel("mpg prediksi"); plt.title("Aktual vs Prediksi (data uji)")
plt.tight_layout()
plt.savefig("static/actual_vs_pred.png", dpi=120)
plt.close()

# 5. Simpan model + metadata -------------------------------------------------
joblib.dump(model, "model/model_mpg.joblib")
meta = {
    "intercept": float(model.intercept_),
    "coef": float(model.coef_[0]),
    "r2": float(r2), "rmse": rmse, "mae": float(mae),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "weight_min": float(X[FEATURE].min()), "weight_max": float(X[FEATURE].max()),
    "correlation": float(corr),
}
with open("model/metrics.json", "w") as f:
    json.dump(meta, f, indent=2)
print("\nModel & metrik tersimpan di folder model/")
