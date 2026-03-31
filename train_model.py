"""Train and persist a simple risk-classification model for EMI assessments."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


RANDOM_SEED = 42
SAMPLE_SIZE = 5000
MODEL_FILE = Path(__file__).resolve().parent / "model.pkl"


def calculate_monthly_emi(loan_amount: np.ndarray, tenure: np.ndarray, annual_rate: float = 12.0) -> np.ndarray:
    """Compute monthly EMI with a fixed annual interest rate."""

    monthly_rate = annual_rate / (12 * 100)
    growth = (1 + monthly_rate) ** tenure
    return loan_amount * monthly_rate * growth / (growth - 1)


def generate_synthetic_dataset(sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate synthetic financial samples and derive risk labels."""

    rng = np.random.default_rng(seed)

    income = rng.uniform(30_000, 250_000, sample_size)
    loan_amount = rng.uniform(150_000, 5_000_000, sample_size)
    liabilities = rng.uniform(0, 90_000, sample_size)
    tenure = rng.integers(12, 121, sample_size)

    monthly_emi = calculate_monthly_emi(loan_amount=loan_amount, tenure=tenure)
    emi_to_income_ratio = (monthly_emi + liabilities) / np.maximum(income, 1)

    risk = np.select(
        [emi_to_income_ratio <= 0.35, emi_to_income_ratio <= 0.55],
        ["Low", "Medium"],
        default="High",
    )

    return pd.DataFrame(
        {
            "income": income,
            "loan_amount": loan_amount,
            "liabilities": liabilities,
            "tenure": tenure,
            "risk": risk,
        }
    )


def train_risk_model(dataframe: pd.DataFrame) -> RandomForestClassifier:
    """Train a random-forest classifier on synthetic financial features."""

    feature_columns = ["income", "loan_amount", "liabilities", "tenure"]
    features = dataframe[feature_columns]
    target = dataframe["risk"]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=RANDOM_SEED,
    )
    model.fit(features, target)
    return model


def save_model(model: RandomForestClassifier, output_path: Path = MODEL_FILE) -> None:
    """Persist the trained model to disk."""

    joblib.dump(model, output_path)


def main() -> None:
    """Train and save the risk model artifact."""

    dataset = generate_synthetic_dataset()
    model = train_risk_model(dataset)
    save_model(model)
    print(f"Risk model saved at: {MODEL_FILE}")


if __name__ == "__main__":
    main()
