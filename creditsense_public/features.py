from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from .paths import resolve_data_file


SEED = 42
USE_POLYNOMIAL_FEATURES = True
MAX_POLY_BASE_COLS = 18


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def add_semantic_missingness_features(out: pd.DataFrame) -> pd.DataFrame:
    def miss(col: str) -> pd.Series:
        flag = f"{col}_is_missing"
        if flag in out.columns:
            return out[flag].fillna(0).astype(np.int8)
        return pd.Series(0, index=out.index, dtype=np.int8)

    home = (
        out["HomeOwnership"].fillna("__MISSING__").astype(str).str.lower()
        if "HomeOwnership" in out.columns
        else pd.Series("__MISSING__", index=out.index)
    )
    education = (
        out["EducationLevel"].fillna("__MISSING__").astype(str).str.lower()
        if "EducationLevel" in out.columns
        else pd.Series("__MISSING__", index=out.index)
    )
    loan_purpose = (
        out["LoanPurpose"].fillna("__MISSING__").astype(str).str.lower()
        if "LoanPurpose" in out.columns
        else pd.Series("__MISSING__", index=out.index)
    )

    renter_like = home.isin(["rent", "living with family"])
    owner_like = home.isin(["own", "mortgage"])
    younger_or_school = (
        (out["Age"].fillna(0) <= 32)
        | education.isin(["some college", "bachelor", "master", "phd"])
        | (loan_purpose == "education")
    )

    property_missing = miss("PropertyValue")
    mortgage_missing = miss("MortgageOutstandingBalance")
    student_missing = miss("StudentLoanOutstandingBalance")
    collateral_type_missing = miss("CollateralType")
    collateral_value_missing = miss("CollateralValue")
    vehicle_missing = miss("VehicleValue")
    auto_loan_missing = miss("AutoLoanOutstandingBalance")
    secondary_income_missing = miss("SecondaryMonthlyIncome")

    out["PropertyMissingRentalFlag"] = (renter_like & (property_missing == 1)).astype(np.int8)
    out["PropertyPresentNonOwnerFlag"] = (renter_like & (property_missing == 0)).astype(np.int8)
    out["MortgageMissingOwnerFlag"] = (owner_like & (mortgage_missing == 1)).astype(np.int8)
    out["MortgagePresentNonOwnerFlag"] = (renter_like & (mortgage_missing == 0)).astype(np.int8)
    out["StudentLoanMissingYoungEducatedFlag"] = (younger_or_school & (student_missing == 1)).astype(np.int8)
    out["StudentLoanPresentOlderFlag"] = ((out["Age"].fillna(0) >= 40) & (student_missing == 0)).astype(np.int8)
    out["NoCollateralFlag"] = ((collateral_type_missing == 1) & (collateral_value_missing == 1)).astype(np.int8)
    out["CollateralMismatchFlag"] = (collateral_type_missing != collateral_value_missing).astype(np.int8)
    out["VehicleValueNoAutoLoanFlag"] = ((vehicle_missing == 0) & (auto_loan_missing == 1)).astype(np.int8)
    out["AutoLoanNoVehicleFlag"] = ((vehicle_missing == 1) & (auto_loan_missing == 0)).astype(np.int8)
    out["SecondaryIncomeMissingCoApplicantFlag"] = (
        (secondary_income_missing == 1) & (out["HasCoApplicant"].fillna(0) == 1)
    ).astype(np.int8)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    miss_cols = [col for col in out.columns if out[col].isna().any()]
    if miss_cols:
        miss = out[miss_cols].isna().astype(np.int8)
        miss.columns = [f"{col}_is_missing" for col in miss_cols]
        out = pd.concat([out, miss], axis=1)

    out["LatePaymentTotal"] = (
        out["NumberOfLatePayments30Days"].fillna(0)
        + out["NumberOfLatePayments60Days"].fillna(0)
        + out["NumberOfLatePayments90Days"].fillna(0)
    )
    out["LatePaymentSeverity"] = (
        out["NumberOfLatePayments30Days"].fillna(0)
        + 2 * out["NumberOfLatePayments60Days"].fillna(0)
        + 3 * out["NumberOfLatePayments90Days"].fillna(0)
    )
    out["TotalOutstandingDebt"] = (
        out["MortgageOutstandingBalance"].fillna(0)
        + out["AutoLoanOutstandingBalance"].fillna(0)
        + out["StudentLoanOutstandingBalance"].fillna(0)
    )
    out["NetWorthProxy"] = out["TotalAssets"].fillna(0) - out["TotalOutstandingDebt"]

    out["IncomePerDependent"] = safe_div(out["TotalMonthlyIncome"].fillna(0), out["NumberOfDependents"].fillna(0) + 1)
    out["IncomeToLoanRatio"] = safe_div(out["AnnualIncome"], out["RequestedLoanAmount"])
    out["LoanShareOfIncome"] = safe_div(out["RequestedLoanAmount"], out["AnnualIncome"])
    out["MonthlyPaymentShare"] = safe_div(out["MonthlyPaymentEstimate"], out["TotalMonthlyIncome"])
    out["DebtPressure"] = (
        out["DebtToIncomeRatio"].fillna(0)
        + out["PaymentToIncomeRatio"].fillna(0)
        + out["LoanToIncomeRatio"].fillna(0)
    )
    out["AvailableCreditProxy"] = out["TotalCreditLimit"].fillna(0) * (1 - out["RevolvingUtilizationRate"].fillna(0))
    out["CreditHeadroom"] = out["TotalCreditLimit"].fillna(0) - out["TotalOutstandingDebt"].fillna(0)
    out["DebtToAssets"] = safe_div(out["TotalOutstandingDebt"], out["TotalAssets"])
    out["AssetCoverage"] = safe_div(out["TotalAssets"], out["RequestedLoanAmount"])
    out["LiquidityBuffer"] = safe_div(
        out["SavingsBalance"].fillna(0) + out["CheckingBalance"].fillna(0),
        out["MonthlyPaymentEstimate"].fillna(0) + 1,
    )
    out["InquiryAcceleration"] = out["NumberOfHardInquiries24Mo"].fillna(0) - out["NumberOfHardInquiries12Mo"].fillna(0)
    out["RiskyComboFlag"] = (
        (out["DebtPressure"].fillna(0) > 1.0) & (out["LatePaymentSeverity"].fillna(0) > 0)
    ).astype(np.int8)

    out = add_semantic_missingness_features(out)

    out["DerogatoryCount"] = (
        out["NumberOfChargeOffs"].fillna(0)
        + out["NumberOfCollections"].fillna(0)
        + out["NumberOfBankruptcies"].fillna(0)
        + out["NumberOfPublicRecords"].fillna(0)
    )
    out["DerogatorySeverity"] = (
        2.5 * out["NumberOfChargeOffs"].fillna(0)
        + 1.5 * out["NumberOfCollections"].fillna(0)
        + 4.0 * out["NumberOfBankruptcies"].fillna(0)
        + 1.0 * out["NumberOfPublicRecords"].fillna(0)
    )
    out["LatePaymentPerOpenAccount"] = safe_div(out["LatePaymentSeverity"].fillna(0), out["NumberOfOpenAccounts"].fillna(0) + 1)
    out["LatePaymentPerHistoryYear"] = safe_div(out["LatePaymentSeverity"].fillna(0), (out["CreditHistoryLengthMonths"].fillna(0) / 12.0) + 1)
    out["DerogatoryPerHistoryYear"] = safe_div(out["DerogatorySeverity"].fillna(0), (out["CreditHistoryLengthMonths"].fillna(0) / 12.0) + 1)
    out["InquiryPerHistoryYear"] = safe_div(out["NumberOfHardInquiries24Mo"].fillna(0), (out["CreditHistoryLengthMonths"].fillna(0) / 12.0) + 1)
    out["UtilizationXDebtToIncome"] = out["RevolvingUtilizationRate"].fillna(0) * out["DebtToIncomeRatio"].fillna(0)
    out["UtilizationXPaymentStress"] = out["RevolvingUtilizationRate"].fillna(0) * out["PaymentToIncomeRatio"].fillna(0)
    out["LateSeverityXUtilization"] = out["LatePaymentSeverity"].fillna(0) * out["RevolvingUtilizationRate"].fillna(0)
    out["LateSeverityXDebtToIncome"] = out["LatePaymentSeverity"].fillna(0) * out["DebtToIncomeRatio"].fillna(0)
    out["DerogatoryXUtilization"] = out["DerogatorySeverity"].fillna(0) * out["RevolvingUtilizationRate"].fillna(0)
    out["DerogatoryXDebtPressure"] = out["DerogatorySeverity"].fillna(0) * out["DebtPressure"].fillna(0)
    out["ThinFileStress"] = (
        ((out["CreditHistoryLengthMonths"].fillna(0) < 24) | (out["NumberOfOpenAccounts"].fillna(0) < 2)).astype(np.int8)
        * (1.0 + out["RevolvingUtilizationRate"].fillna(0) + out["LatePaymentSeverity"].fillna(0))
    )
    out["SeasonedStressFlag"] = (
        (out["CreditHistoryLengthMonths"].fillna(0) >= 120)
        & (
            (out["RevolvingUtilizationRate"].fillna(0) > 0.65)
            | (out["LatePaymentSeverity"].fillna(0) > 0)
            | (out["DerogatoryCount"].fillna(0) > 0)
        )
    ).astype(np.int8)

    money_cols = [
        "RequestedLoanAmount",
        "AnnualIncome",
        "TotalAssets",
        "TotalOutstandingDebt",
        "TotalCreditLimit",
        "MonthlyPaymentEstimate",
        "PropertyValue",
        "SavingsBalance",
        "CheckingBalance",
        "InvestmentPortfolioValue",
        "CollateralValue",
    ]
    for col in money_cols:
        if col in out.columns:
            out[f"log1p_{col}"] = np.log1p(np.clip(out[col].fillna(0), 0, None))

    return out


def choose_poly_columns(df: pd.DataFrame, max_cols: int = MAX_POLY_BASE_COLS) -> list[str]:
    preferred = [
        "DebtPressure",
        "LoanShareOfIncome",
        "MonthlyPaymentShare",
        "DebtToIncomeRatio",
        "PaymentToIncomeRatio",
        "LoanToIncomeRatio",
        "RevolvingUtilizationRate",
        "CreditHeadroom",
        "AvailableCreditProxy",
        "TotalOutstandingDebt",
        "AnnualIncome",
        "TotalMonthlyIncome",
        "RequestedLoanAmount",
        "NetWorthProxy",
        "AssetCoverage",
        "DebtToAssets",
        "LatePaymentTotal",
        "LatePaymentSeverity",
        "NumberOfHardInquiries24Mo",
        "IncomeToLoanRatio",
    ]
    return [col for col in preferred if col in df.columns][:max_cols]


def add_polynomial_features(df: pd.DataFrame, poly_cols: list[str]) -> pd.DataFrame:
    if not poly_cols:
        return df
    out = df.copy()
    fill = out[poly_cols].fillna(0)
    poly_data: Dict[str, pd.Series] = {}
    for col in poly_cols:
        poly_data[f"poly2_{col}"] = fill[col] * fill[col]
    for i, left in enumerate(poly_cols):
        left_vals = fill[left]
        for right in poly_cols[i + 1 :]:
            poly_data[f"poly_{left}_x_{right}"] = left_vals * fill[right]
    if poly_data:
        out = pd.concat([out, pd.DataFrame(poly_data, index=out.index)], axis=1)
    return out


@dataclass
class Preprocessor:
    use_poly: bool
    poly_cols: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    medians: pd.Series
    encoder: OrdinalEncoder
    freq_maps: dict[str, dict[str, float]]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = add_features(frame)
        for col in self.cat_cols:
            df[f"{col}_freq"] = df[col].fillna("__MISSING__").astype(str).map(self.freq_maps[col]).fillna(0.0)
        if self.use_poly:
            df = add_polynomial_features(df, self.poly_cols)
        for col in self.num_cols:
            if col not in df.columns:
                df[col] = np.nan
        for col in self.cat_cols:
            if col not in df.columns:
                df[col] = "__MISSING__"
        num = df[self.num_cols].copy().fillna(self.medians)
        cat = df[self.cat_cols].copy().fillna("__MISSING__").astype(str)
        cat_enc = pd.DataFrame(self.encoder.transform(cat), columns=self.cat_cols, index=df.index)
        return pd.concat([num, cat_enc], axis=1).fillna(0.0)


def fit_preprocessor(frame: pd.DataFrame, use_poly: bool = USE_POLYNOMIAL_FEATURES) -> Preprocessor:
    df = add_features(frame)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    freq_maps: dict[str, dict[str, float]] = {}
    for col in cat_cols:
        vc = df[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
        freq_maps[col] = vc.to_dict()
        df[f"{col}_freq"] = df[col].fillna("__MISSING__").astype(str).map(freq_maps[col])

    poly_cols = choose_poly_columns(df) if use_poly else []
    if use_poly:
        df = add_polynomial_features(df, poly_cols)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    medians = df[num_cols].median()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(df[cat_cols].fillna("__MISSING__").astype(str))

    return Preprocessor(
        use_poly=use_poly,
        poly_cols=poly_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        medians=medians,
        encoder=encoder,
        freq_maps=freq_maps,
    )


def load_competition_data() -> Dict[str, pd.DataFrame | pd.Series]:
    train_df = pd.read_csv(resolve_data_file("credit_train.csv"))
    test_df = pd.read_csv(resolve_data_file("credit_test.csv"))
    X = train_df.drop(columns=["RiskTier", "InterestRate"])
    y_cls = train_df["RiskTier"].astype(int)
    y_reg = train_df["InterestRate"].astype(float)
    return {
        "train_df": train_df,
        "test_df": test_df,
        "X": X,
        "y_cls": y_cls,
        "y_reg": y_reg,
    }


def split_train_validation(X: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series) -> tuple:
    return train_test_split(
        X,
        y_cls,
        y_reg,
        test_size=0.2,
        random_state=SEED,
        stratify=y_cls,
    )
