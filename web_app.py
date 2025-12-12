import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime

# --------------------------------------------------
# Helper functions for feature engineering
# (mirror what you did in the notebook as closely
# as possible; these are reasonable defaults)
# --------------------------------------------------

def part_of_day(h):
    """Return Morning / Afternoon / Evening / Night based on hour."""
    if pd.isna(h):
        return "Unknown"
    h = int(h)
    if 0 <= h < 6:
        return "Night"
    elif 6 <= h < 12:
        return "Morning"
    elif 12 <= h < 18:
        return "Afternoon"
    else:
        return "Evening"


def age_bucket(age):
    """Bucketize age (tweak if your notebook used different buckets)."""
    age = float(age)
    if age < 25:
        return "Young"
    elif age < 45:
        return "Adult"
    elif age < 65:
        return "Middle_Aged"
    else:
        return "Senior"


def simplify_location(loc):
    """Simplify location into fewer categories."""
    if not loc:
        return "Other"
    loc = loc.strip()
    # Example: keep some special locations; else 'Other'
    top = ["New York", "Los Angeles", "Chicago", "New Jersey"]
    if loc in top:
        return loc
    return "Other"


def is_new_account(account_age_days, threshold=180):
    """Return 1 if account is 'new' based on days; adjusting threshold if needed in the future."""
    return 1 if float(account_age_days) < threshold else 0


# --------------------------------------------------
# Flask app setup
# --------------------------------------------------

app = Flask(__name__)

# Load the trained pipeline (preprocessor + model)
# Run this file from the project root:
#   python src/web_app.py
model = joblib.load("models/fraud_model.pkl")

# Optional: overall test accuracy of your model
MODEL_ACCURACY = 0.93  # change to your real test accuracy if you want


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None
    error_msg = None

    if request.method == "POST":
        try:
            # ---- 1. Read numeric inputs ----
            amount = float(request.form["amount"])
            quantity = float(request.form["quantity"])
            age = float(request.form["age"])
            account_age = float(request.form["account_age"])
            hour = float(request.form["hour"])

            # ---- 2. Read transaction date ----
            date_str = request.form["trans_date"]  # YYYY-MM-DD
            # Parse safely
            trans_dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_of_week = trans_dt.weekday()  # 0=Monday, 6=Sunday

            # ---- 3. Read categorical inputs ----
            payment_method = request.form["payment_method"]
            product_category = request.form["product_category"]
            device_used = request.form["device_used"]
            customer_location = request.form["customer_location"]

            # ---- 4. Build base row with raw columns ----
            row = pd.DataFrame([{
                "Transaction Amount": amount,
                "Quantity": quantity,
                "Customer Age": age,
                "Account Age Days": account_age,
                "Transaction Hour": hour,
                "Payment Method": payment_method,
                "Product Category": product_category,
                "Device Used": device_used,
                "Customer Location": customer_location,
            }])

            # ---- 5. Add engineered features expected by the model ----

            # Day-of-week and weekend
            row["Trans_DayOfWeek"] = day_of_week
            row["Trans_IsWeekend"] = 1 if day_of_week >= 5 else 0

            # Part of day
            row["Trans_PartOfDay"] = part_of_day(hour)

            # Log amount (log1p to avoid log(0))
            row["Log_TransactionAmount"] = np.log1p(amount)

            # New account flag
            row["Is_New_Account"] = is_new_account(account_age)

            # Age bucket
            row["Customer_Age_Bucket"] = age_bucket(age)

            # Short location category
            row["Customer_Location_Short"] = simplify_location(customer_location)

            # ---- 6. Predict with the pipeline ----
            proba = model.predict_proba(row)[0][1]  # probability of class "1" (fraud)
            prob = round(proba * 100, 2)
            prediction = "Fraudulent" if proba >= 0.5 else "Not Fraudulent"

        except Exception as e:
            prediction = "Error"
            prob = 0
            error_msg = str(e)
            print("Error while predicting:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        prob=prob,
        model_accuracy=int(MODEL_ACCURACY * 100),
        error_msg=error_msg,
    )


if __name__ == "__main__":
    app.run(debug=True)
    
