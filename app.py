import json
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model/model_mpg.joblib")
with open("model/metrics.json") as f:
    META = json.load(f)

KG_TO_LBS = 2.20462
MPG_TO_KML = 0.425144  # 1 mpg (US) = 0.425144 km/l


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, form = None, None, {"weight": "", "unit": "kg"}
    if request.method == "POST":
        form["weight"] = request.form.get("weight", "").strip()
        form["unit"] = request.form.get("unit", "kg")
        try:
            value = float(form["weight"].replace(",", "."))
            if value <= 0:
                raise ValueError
            lbs = value * KG_TO_LBS if form["unit"] == "kg" else value
            mpg = float(model.predict(pd.DataFrame({"weight": [lbs]}))[0])
            in_range = META["weight_min"] <= lbs <= META["weight_max"]
            result = {
                "lbs": round(lbs),
                "kg": round(lbs / KG_TO_LBS),
                "mpg": round(mpg, 1),
                "kml": round(mpg * MPG_TO_KML, 1),
                "in_range": in_range,
                "negative": mpg <= 0,
            }
        except ValueError:
            error = "Masukkan bobot kendaraan berupa angka lebih dari 0."
    return render_template("index.html", result=result, error=error, form=form, meta=META)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
