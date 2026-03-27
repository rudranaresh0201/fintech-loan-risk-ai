"""API tests for finance endpoints."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.services import prediction_service


class ApiDummyScaler:
    """Mock scaler for API tests."""

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy(dtype=float)


class ApiDummyClassifier:
    """Mock classifier with deterministic outputs."""

    coef_ = np.array([[0.4, 0.1, -0.2, 0.3]])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([0])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return np.array([[0.25, 0.75]])


class ApiDummyRegressor:
    """Mock regressor with deterministic outputs."""

    coef_ = np.array([0.3, 0.2, 0.1, -0.1])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.array([12000.0])


def test_post_predict_success(
    client,
    monkeypatch,
    valid_financial_profile_payload: dict,
) -> None:
    """POST /predict should return a valid prediction payload."""

    feature_columns = ["f1", "f2", "f3", "f4"]

    def fake_load_artifacts() -> SimpleNamespace:
        return SimpleNamespace(
            classifier=ApiDummyClassifier(),
            regressor=ApiDummyRegressor(),
            scaler=ApiDummyScaler(),
            feature_columns=feature_columns,
        )

    def fake_build_model_input(payload, feature_columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame([[1.0, 2.0, 1.0, 0.5]], columns=feature_columns)

    monkeypatch.setattr(prediction_service, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(prediction_service, "build_model_input", fake_build_model_input)

    response = client.post("/predict", json=valid_financial_profile_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligibility_label"] == "Eligible"
    assert payload["max_monthly_emi_predicted"] == 12000.0
    assert payload["confidence"] == 0.75
    assert "explainability" in payload


def test_post_predict_invalid_credit_score(client, valid_financial_profile_payload: dict) -> None:
    """POST /predict should reject schema-invalid credit score."""

    bad_payload = dict(valid_financial_profile_payload)
    bad_payload["credit_score"] = 200

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_post_predict_invalid_dependents(client, valid_financial_profile_payload: dict) -> None:
    """POST /predict should reject cross-field invalid dependents."""

    bad_payload = dict(valid_financial_profile_payload)
    bad_payload["dependents"] = bad_payload["family_size"] + 1

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_post_calculate_emi_success(client, valid_emi_payload: dict) -> None:
    """POST /calculate-emi should return EMI summary fields."""

    response = client.post("/calculate-emi", json=valid_emi_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["monthly_emi"] > 0
    assert payload["total_payable"] > valid_emi_payload["principal"]
    assert payload["total_interest"] > 0


def test_post_calculate_emi_invalid_principal(client, valid_emi_payload: dict) -> None:
    """POST /calculate-emi should reject negative principal."""

    bad_payload = dict(valid_emi_payload)
    bad_payload["principal"] = -1

    response = client.post("/calculate-emi", json=bad_payload)

    assert response.status_code == 422
