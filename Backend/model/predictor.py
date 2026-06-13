import os
import joblib
import pandas as pd


class ChurnPredictor:

    def __init__(self):

        BASE_DIR = os.path.dirname(
            os.path.abspath(__file__)
        )

        model_path = os.path.join(
            BASE_DIR,
            "churn_model.pkl"
        )

        artifacts = joblib.load(
            model_path
        )

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

        scaled = self.scaler.transform(
            input_df
        )

        probability = float(
            self.model.predict_proba(
                scaled
            )[0][1]
        )

        risk_level = (
            "Low"
            if probability < 0.35
            else "Medium"
            if probability < 0.70
            else "High"
        )

        # Risk Drivers
        reasons = []

        if payload["SatisfactionScore"] <= 2:
            reasons.append(
                "Low Satisfaction Score"
            )

        if payload["Contract"] == "Month-to-Month":
            reasons.append(
                "Month-to-Month Contract"
            )

        if payload["TenureinMonths"] < 12:
            reasons.append(
                "Short Customer Tenure"
            )

        if payload["MonthlyCharge"] > 80:
            reasons.append(
                "High Monthly Charges"
            )

        if payload["InternetType"] == "Fiber Optic":
            reasons.append(
                "Fiber Optic Internet"
            )

        if len(reasons) == 0:
            reasons.append(
                "No significant risk drivers"
            )

        # Customer Persona
        if (
            probability > 0.70
            and
            payload["MonthlyCharge"] > 80
        ):

            persona = (
                "At-Risk Premium Customer"
            )

        elif probability > 0.70:

            persona = (
                "At-Risk Customer"
            )

        elif (
            probability < 0.30
            and
            payload["TenureinMonths"] > 24
        ):

            persona = (
                "Loyal Customer"
            )

        else:

            persona = (
                "Regular Customer"
            )

        # Revenue At Risk
        revenue_at_risk = round(
            payload["MonthlyCharge"]
            * 12
            * probability,
            2
        )

        # Recommendations
        recommendations = []

        if probability > 0.70:

            recommendations.append(
                "Offer retention discount"
            )

        if payload["Contract"] == "Month-to-Month":

            recommendations.append(
                "Upgrade to annual contract"
            )

        if payload["SatisfactionScore"] <= 2:

            recommendations.append(
                "Assign customer success specialist"
            )

        if len(recommendations) == 0:

            recommendations.append(
                "No action required"
            )

        return {

            "churn_probability": round(
                probability * 100,
                2
            ),

            "risk_level": risk_level,

            "top_risk_drivers": reasons,

            "persona": persona,

            "revenue_at_risk": revenue_at_risk,

            "recommendations": recommendations

        }