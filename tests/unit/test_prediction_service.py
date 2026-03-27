"""Unit tests for prediction service with mocked model artifacts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.services import prediction_service
from schemas.requests import FinancialProfile


class DummyScaler:
    """Simple scaler mock returning deterministic numeric arrays."""

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=float)


class DummyClassifier:
    """Classifier mock supporting predict and predict_proba."""

    coef_ = np.array([[0.4, -0.2, 0.1, 0.3]])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([0])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.array([[0.1, 0.9]])


class DummyRegressor:
    """Regressor mock with linear coefficients."""

    coef_ = np.array([0.2, 0.1, -0.3, 0.4])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([15555.678])


class DummyClassifierNoProba:
    """Classifier mock without probability API."""

    coef_ = np.array([[0.2, 0.2, 0.2, 0.2]])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([1])


class DummyRegressorNegative:
    """Regressor mock returning a negative prediction for clamping test."""

    coef_ = np.array([0.1, 0.1, 0.1, 0.1])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([-5000.0])


def test_predict_emi_eligibility_returns_structured_response(
    monkeypatch, valid_financial_profile_payload: dict
) -> None:
    """Service should return stable response with confidence and explainability."""

    profile = FinancialProfile(**valid_financial_profile_payload)

    feature_columns = ["f1", "f2", "f3", "f4"]

    def fake_build_model_input(
        payload: FinancialProfile, feature_columns: list[str]
    ) -> pd.DataFrame:
        return pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=feature_columns)

    def fake_load_artifacts() -> SimpleNamespace:
        return SimpleNamespace(
            classifier=DummyClassifier(),
            regressor=DummyRegressor(),
            scaler=DummyScaler(),
            feature_columns=feature_columns,
        )

    monkeypatch.setattr(prediction_service, "build_model_input", fake_build_model_input)
    monkeypatch.setattr(prediction_service, "load_artifacts", fake_load_artifacts)

    response = prediction_service.predict_emi_eligibility(profile)

    assert response.eligibility_code == 0
    assert response.eligibility_label == "Eligible"
    assert response.max_monthly_emi_predicted == 15555.68
    assert response.confidence == 0.9
    assert response.formula_monthly_emi_for_requested_loan > 0
    assert len(response.explainability.rule_based_insights) > 0
    assert len(response.explainability.classification_top_contributors) > 0
    assert len(response.explainability.regression_top_contributors) > 0


def test_predict_emi_eligibility_handles_no_proba_and_negative_regression(
    monkeypatch, valid_financial_profile_payload: dict
) -> None:
    """Confidence should be None and negative regression output should be clamped."""

    profile = FinancialProfile(**valid_financial_profile_payload)

    feature_columns = ["f1", "f2", "f3", "f4"]

    def fake_build_model_input(
        payload: FinancialProfile, feature_columns: list[str]
    ) -> pd.DataFrame:
        return pd.DataFrame([[2.0, 2.0, 2.0, 2.0]], columns=feature_columns)

    def fake_load_artifacts() -> SimpleNamespace:
        return SimpleNamespace(
            classifier=DummyClassifierNoProba(),
            regressor=DummyRegressorNegative(),
            scaler=DummyScaler(),
            feature_columns=feature_columns,
        )

    monkeypatch.setattr(prediction_service, "build_model_input", fake_build_model_input)
    monkeypatch.setattr(prediction_service, "load_artifacts", fake_load_artifacts)

    response = prediction_service.predict_emi_eligibility(profile)

    assert response.eligibility_code == 1
    assert response.eligibility_label == "Not Eligible"
    assert response.max_monthly_emi_predicted == 0.0
    assert response.confidence is None
