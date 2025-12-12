import joblib
import numpy as np
import pandas as pd

from datetime import datetime

# Load the trained pipeline
model = joblib.load("models/fraud_model.pkl")

MODEL_ACCURACY = 0.80  

def compute_time_features(date_str: str, hour: int):  # Compute day of week, weekend flag, and part of day
    """
    date_str: 'YYYY-MM-DD' (e.g. '2025-11-30')
    hour: 0-23
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # fallback if user gives bad date
        dt = None

    if dt is not None:
        day_of_week = dt.weekday()   # 0=Mon, 6=Sun
    else:
        day_of_week = 0  # default Monday

    is_weekend = int(day_of_week >= 5) 

    # part of day
    if 0 <= hour < 6:
        part = "Night"
    elif 6 <= hour < 12:
        part = "Morning"
    elif 12 <= hour < 18:
        part = "Afternoon"
    else:
        part = "Evening"

    return day_of_week, is_weekend, part

def age_bucket(age: int) -> str:
    if age < 20:
        return "<20"
    elif age < 30:
        return "20-29"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    else:
        return "60+"
    
def build_transaction_df(): # Build a DataFrame for a single transaction from user input
    print("\nPlease enter transaction details.\n(Press Ctrl+C to quit anytime.)\n")

    amount = float(input("Transaction amount: "))
    quantity = int(input("Quantity: "))
    cust_age = int(input("Customer age: "))
    acc_age = int(input("Account age in days: "))

    date_str = input("Transaction date (YYYY-MM-DD): ")
    hour = int(input("Transaction hour (0-23): "))

    payment_method = input("Payment method (e.g., Credit Card, PayPal): ")
    product_category = input("Product category (e.g., Electronics): ")
    device_used = input("Device used (e.g., Mobile, Desktop): ")
    location_short = input("Customer location group (e.g., New York, Other): ")

    # --- engineered features ---
    log_amount = np.log1p(amount)
    is_new_account = int(acc_age < 30)
    day_of_week, is_weekend, part_of_day = compute_time_features(date_str, hour)
    age_group = age_bucket(cust_age)

    # build DataFrame with ONE row
    row = {
        "Transaction Amount": amount,
        "Log_TransactionAmount": log_amount,
        "Quantity": quantity,
        "Customer Age": cust_age,
        "Account Age Days": acc_age,
        "Is_New_Account": is_new_account,
        "Transaction Hour": hour,
        "Trans_DayOfWeek": day_of_week,
        "Trans_IsWeekend": is_weekend,
        "Payment Method": payment_method,
        "Product Category": product_category,
        "Device Used": device_used,
        "Customer_Location_Short": location_short,
        "Customer_Age_Bucket": age_group,
        "Trans_PartOfDay": part_of_day,
    }

    return pd.DataFrame([row])

def explain_prediction(row: pd.Series, prob: float, pred: int) -> str: # Explain the prediction with key risk factors
    reasons = []

    amount = row["Transaction Amount"]
    acc_age = row["Account Age Days"]
    part_of_day = row["Trans_PartOfDay"]
    is_weekend = row["Trans_IsWeekend"]
    device = row["Device Used"]

    if amount > 1000:
        reasons.append("high transaction amount")
    if acc_age < 30:
        reasons.append("very new account")
    if part_of_day == "Night":
        reasons.append("night-time transaction")
    if is_weekend == 1:
        reasons.append("weekend transaction")
    if isinstance(device, str) and device.lower() == "mobile":
        reasons.append("transaction made on mobile")

    base = f"Estimated fraud probability: {prob:.2%}. " # Base fraud probability message

    if pred == 1:
        verdict = "This transaction is LIKELY FRAUDULENT."
    else:
        verdict = "This transaction is likely NOT fraudulent."

    if reasons:
        reason_text = " Key factors: " + ", ".join(reasons) + "."
    else:
        reason_text = " No strong risk factors were detected."

    return base + verdict + reason_text

def chat(): # Main chatbot loop for user interaction
    print("🤖 Fraud Detection Assistant")
    print("---------------------------")
    print(f"Model test accuracy (overall): {MODEL_ACCURACY:.2f}")
    print("I will ask you for transaction details and estimate fraud risk.\n")

    while True: # Main interaction loop
        try:
            tx_df = build_transaction_df() # ask user questions, amount, age, device and give 1 row dataframe
            row = tx_df.iloc[0] # get the single row as series and use for later explanation

            # model prediction
            proba = model.predict_proba(tx_df)[0][1]   # probability of class 1 (fraud)
            pred = model.predict(tx_df)[0] # predicted class 0/1

            print("\n=== Prediction ===")
            explanation = explain_prediction(row, proba, pred)
            print(explanation)

            again = input("\nDo you want to check another transaction? (y/n): ").strip().lower()
            if again != "y":
                print("Goodbye! 👋")
                break

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nSomething went wrong: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != "y":
                break


if __name__ == "__main__":
    chat()