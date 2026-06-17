"""
CreditScorePreprocessor: encapsulates every cleaning / imputation / feature
engineering / encoding / scaling step validated in exploration.ipynb, behind
a scikit-learn-style fit/transform API (same idiom as the lecturer's
DataProcessor in OOP Dosen/direct_association.ipynb).

All parameters learned from data (medians, modes, the Credit_Mix
conditional-imputation lookup, the one-hot column layout, the scaler) are
private attributes set during fit() and only ever fit on the train set, then
replayed identically by transform() on the test set or a single inference
row -- so this one class is reused unchanged by pipeline.py and inference.py.
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

TARGET_COL = "Credit_Score"
TARGET_MAP = {"Poor": 0, "Standard": 1, "Good": 2}

IRRELEVANT_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]

COLS_ABS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly",
]
COLS_ALLOW_NEGATIVE = ["Changed_Credit_Limit", "Monthly_Balance", "Delay_from_due_date"]

NAN_SENTINELS = {
    "Occupation": "_______",
    "Credit_Mix": "_",
    "Payment_Behaviour": "!@9#%8",
}

DIRTY_OUTLIER_COLS = [
    "Age", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
]

MONTHLY_BALANCE_SENTINEL_THRESHOLD = 1e6

NUM_IMPUTE_COLS = [
    "Monthly_Inhand_Salary", "Credit_History_Age", "Num_of_Delayed_Payment",
    "Amount_invested_monthly", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Monthly_Balance",
]
MODE_IMPUTE_COLS = ["Occupation", "Payment_Behaviour"]

DELAY_BINS = [-1, 0, 2, 5, np.inf]
DELAY_LABELS = ["No Delay", "Low Delay", "Medium Delay", "High Delay"]

FEATURE_SELECTED_OUT = ["Annual_Income"]

NOMINAL_ONEHOT_COLS = ["Month", "Occupation", "Payment_of_Min_Amount", "Payment_Behaviour"]
CREDIT_MIX_ORDER = {"Bad": 0, "Standard": 1, "Good": 2}

NUMERIC_COLS_TO_SCALE = [
    "Age", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance",
]


def _clean_numeric(series, allow_negative=False):
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = pd.to_numeric(cleaned, errors="coerce")
    if not allow_negative:
        cleaned = cleaned.abs()
    return cleaned


def _parse_credit_history_age(val):
    if pd.isna(val):
        return np.nan
    years = re.search(r"(\d+)\s*Year", str(val))
    months = re.search(r"(\d+)\s*Month", str(val))
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return y * 12 + m


def _parse_loan_types(series):
    def split_loans(val):
        if pd.isna(val):
            return []
        parts = re.split(r",\s*|\s+and\s+", str(val))
        return [p.strip() for p in parts if p.strip() and p.strip().lower() != "nan"]
    return series.apply(split_loans)


class CreditScorePreprocessor:
    def __init__(self):
        self.__is_fitted = False
        self.__all_loan_types = None
        self.__outlier_caps = None
        self.__medians = None
        self.__modes = None
        self.__credit_mix_mode_by_bin = None
        self.__credit_mix_fallback_mode = None
        self.__feature_columns = None
        self.__scaler = None

    @property
    def is_fitted(self):
        """Read-only getter -- mirrors EncapsulatedRegressor's guarded private state."""
        return self.__is_fitted

    # ------------------------------------------------------------------
    # Steps shared by fit() and transform() that don't need fitted state
    # ------------------------------------------------------------------
    def _drop_irrelevant(self, df):
        return df.drop(columns=[c for c in IRRELEVANT_COLS if c in df.columns])

    def _clean_numeric_columns(self, df):
        df = df.copy()
        for col in COLS_ABS:
            if col in df.columns:
                df[col] = _clean_numeric(df[col], allow_negative=False)
        for col in COLS_ALLOW_NEGATIVE:
            if col in df.columns:
                df[col] = _clean_numeric(df[col], allow_negative=True)
        return df

    def _apply_nan_sentinels(self, df):
        df = df.copy()
        for col, sentinel in NAN_SENTINELS.items():
            df[col] = df[col].replace(sentinel, np.nan)
        return df

    def _convert_credit_history_age(self, df):
        df = df.copy()
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(_parse_credit_history_age)
        return df

    def _encode_loan_types(self, df, loan_types):
        df = df.copy()
        loan_parsed = _parse_loan_types(df["Type_of_Loan"])
        encoded = pd.DataFrame(
            [{f"Loan_{lt.replace(' ', '_')}": int(lt in row) for lt in loan_types} for row in loan_parsed],
            index=df.index,
        )
        df = pd.concat([df.drop(columns=["Type_of_Loan"]), encoded], axis=1)

        # Fix duplicate loan encoding: "X, and Y Loan" left an unmerged
        # "Loan_and_Y_Loan" column distinct from "Loan_Y_Loan" -- same bug
        # found and fixed in exploration.ipynb. Merge via logical OR.
        and_cols = [c for c in df.columns if c.startswith("Loan_and_")]
        for and_col in and_cols:
            base_col = "Loan_" + and_col[len("Loan_and_"):]
            if base_col in df.columns:
                df[base_col] = (df[base_col] | df[and_col]).astype(int)
        df = df.drop(columns=and_cols)
        return df

    def _sanitize_monthly_balance(self, df):
        df = df.copy()
        df.loc[df["Monthly_Balance"].abs() > MONTHLY_BALANCE_SENTINEL_THRESHOLD, "Monthly_Balance"] = np.nan
        return df

    def _drop_annual_income(self, df):
        return df.drop(columns=[c for c in FEATURE_SELECTED_OUT if c in df.columns])

    # ------------------------------------------------------------------
    # fit(): learn every parameter from train data only
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        df = train_df.drop(columns=[TARGET_COL], errors="ignore").copy()

        df = self._drop_irrelevant(df)
        df = self._clean_numeric_columns(df)
        df = self._apply_nan_sentinels(df)
        df = self._convert_credit_history_age(df)

        loan_parsed = _parse_loan_types(df["Type_of_Loan"])
        self.__all_loan_types = sorted(set(lt for sub in loan_parsed for lt in sub))
        df = self._encode_loan_types(df, self.__all_loan_types)

        self.__outlier_caps = {}
        for col in DIRTY_OUTLIER_COLS:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            self.__outlier_caps[col] = q3 + 1.5 * iqr
        for col, upper in self.__outlier_caps.items():
            df[col] = df[col].clip(upper=upper)

        df = self._sanitize_monthly_balance(df)

        self.__medians = {col: df[col].median() for col in NUM_IMPUTE_COLS}
        for col, median_val in self.__medians.items():
            df[col] = df[col].fillna(median_val)

        self.__modes = {col: df[col].mode()[0] for col in MODE_IMPUTE_COLS}
        for col, mode_val in self.__modes.items():
            df[col] = df[col].fillna(mode_val)

        df["Delay_Bin"] = pd.cut(df["Num_of_Delayed_Payment"], bins=DELAY_BINS, labels=DELAY_LABELS)
        self.__credit_mix_mode_by_bin = (
            df[df["Credit_Mix"].notna()]
            .groupby("Delay_Bin", observed=True)["Credit_Mix"]
            .agg(lambda x: x.mode()[0])
        )
        mask = df["Credit_Mix"].isna()
        df.loc[mask, "Credit_Mix"] = df.loc[mask, "Delay_Bin"].map(self.__credit_mix_mode_by_bin)
        self.__credit_mix_fallback_mode = df["Credit_Mix"].mode()[0]
        df["Credit_Mix"] = df["Credit_Mix"].fillna(self.__credit_mix_fallback_mode)
        df = df.drop(columns=["Delay_Bin"])

        df = self._drop_annual_income(df)

        df = pd.get_dummies(df, columns=NOMINAL_ONEHOT_COLS, drop_first=True, dtype=int)
        self.__feature_columns = df.columns.tolist()

        df["Credit_Mix"] = df["Credit_Mix"].map(CREDIT_MIX_ORDER)

        self.__scaler = RobustScaler()
        self.__scaler.fit(df[NUMERIC_COLS_TO_SCALE])

        self.__is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform(): replay the exact same steps using FITTED parameters
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.__is_fitted:
            raise RuntimeError("CreditScorePreprocessor must be fit() before calling transform().")

        df = df.drop(columns=[TARGET_COL], errors="ignore").copy()

        df = self._drop_irrelevant(df)
        df = self._clean_numeric_columns(df)
        df = self._apply_nan_sentinels(df)
        df = self._convert_credit_history_age(df)
        df = self._encode_loan_types(df, self.__all_loan_types)

        for col, upper in self.__outlier_caps.items():
            df[col] = df[col].clip(upper=upper)

        df = self._sanitize_monthly_balance(df)

        for col, median_val in self.__medians.items():
            df[col] = df[col].fillna(median_val)
        for col, mode_val in self.__modes.items():
            df[col] = df[col].fillna(mode_val)

        df["Delay_Bin"] = pd.cut(df["Num_of_Delayed_Payment"], bins=DELAY_BINS, labels=DELAY_LABELS)
        mask = df["Credit_Mix"].isna()
        df.loc[mask, "Credit_Mix"] = df.loc[mask, "Delay_Bin"].map(self.__credit_mix_mode_by_bin)
        df["Credit_Mix"] = df["Credit_Mix"].fillna(self.__credit_mix_fallback_mode)
        df = df.drop(columns=["Delay_Bin"])

        df = self._drop_annual_income(df)

        df = pd.get_dummies(df, columns=NOMINAL_ONEHOT_COLS, drop_first=True, dtype=int)
        df = df.reindex(columns=self.__feature_columns, fill_value=0)

        df["Credit_Mix"] = df["Credit_Mix"].map(CREDIT_MIX_ORDER)

        df[NUMERIC_COLS_TO_SCALE] = self.__scaler.transform(df[NUMERIC_COLS_TO_SCALE])

        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    @staticmethod
    def encode_target(y: pd.Series) -> pd.Series:
        return y.map(TARGET_MAP)
