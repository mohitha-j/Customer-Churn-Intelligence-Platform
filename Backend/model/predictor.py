import os
import joblib
import pandas as pd

class ChurnPredictor:

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(
            BASE_DIR,
            "churn_model.pkl"
        )

        artifacts = joblib.load(model_path)

        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.feature_cols = artifacts["feature_cols"]

    def predict(self, payload):

        input_df = pd.DataFrame(
            0,
            index=[0],
            columns=self.feature_cols
        )

        for col, value in payload.items():

            if col in input_df.columns:
                input_df[col] = value

            dummy_col = f"{col}_{value}"

            if dummy_col in input_df.columns:
                input_df[dummy_col] = 1
        print("\nINPUT SENT TO MODEL")
        print(input_df.T)

        scaled = self.scaler.transform(input_df)

        probability = float(
            self.model.predict_proba(scaled)[0][1]
        )

        risk_level = (
            "Low"
            if probability < 0.35
            else "Medium"
            if probability < 0.70
            else "High"
        )

        reasons = []
        if payload["SatisfactionScore"] <= 2:
            reasons.append("Low Satisfaction Score")
        if payload["Contract"] == "Month-to-Month":
            reasons.append("Month-to-Month Contract")
        if payload["TenureinMonths"] < 12:
            reasons.append("Short Customer Tenure")

        if payload["MonthlyCharge"] > 80:
            reasons.append("High Monthly Charges")

        if payload["InternetType"] == "Fiber Optic":
            reasons.append("Fiber Optic Internet")

        return {
    "churn_probability": round(
        probability * 100,
        2
    ),

    "risk_level": risk_level,

    "top_risk_drivers": reasons[:3]
}