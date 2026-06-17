import os

import joblib
import pandas as pd

from preprocessing import TARGET_MAP

LABEL_DECODE = {v: k for k, v in TARGET_MAP.items()}

class InferenceService:
    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        preprocessor_path = os.path.join("preprocessor.pkl")
        model_path = os.path.join("best_model.pkl")

        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Artifacts not found in '{artifacts_dir}/'. Run pipeline.py first to train and export them."
            )

        self.__preprocessor = joblib.load(preprocessor_path)
        self.__model = joblib.load(model_path)

    def predict(self, raw_input: dict) -> dict:
        input_df = pd.DataFrame([raw_input])
        x = self.__preprocessor.transform(input_df)

        prediction = self.__model.predict(x)[0]
        probabilities = self.__model.predict_proba(x)[0]

        return {
            "credit_score": LABEL_DECODE[int(prediction)],
            "probabilities": {
                LABEL_DECODE[i]: float(prob) for i, prob in enumerate(probabilities)
            },
        }


if __name__ == "__main__":
    service = InferenceService()

    sample_input = {
        "Month": "January",
        "Age": 30,
        "Occupation": "Lawyer",
        "Monthly_Inhand_Salary": 4200.0,
        "Num_Bank_Accounts": 4,
        "Num_Credit_Card": 5,
        "Interest_Rate": 12,
        "Num_of_Loan": 2,
        "Type_of_Loan": "Auto Loan, Personal Loan",
        "Delay_from_due_date": 10,
        "Num_of_Delayed_Payment": 8,
        "Changed_Credit_Limit": 5.5,
        "Num_Credit_Inquiries": 3,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 800.0,
        "Credit_Utilization_Ratio": 30.0,
        "Credit_History_Age": "15 Years and 6 Months",
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 100.0,
        "Amount_invested_monthly": 200.0,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 350.0,
    }

    result = service.predict(sample_input)
    print(result)
