"""API tests for LSTM risk prediction endpoint."""

from __future__ import annotations


def test_post_predict_risk_success(client, monkeypatch) -> None:
    """POST /predict-risk should return risk score and level."""

    from backend.routes import finance
    from schemas.responses import RiskPredictResponse

    def fake_predict_default_risk(payload) -> RiskPredictResponse:
        return RiskPredictResponse(
            risk_score=0.82,
            risk_level="HIGH",
            explanations=["Expenses are growing faster than income trend, increasing default risk."],
        )

    monkeypatch.setattr(finance, "predict_default_risk", fake_predict_default_risk)

    sequence_payload = {
        "sequence": [
            {"income": 100000, "expenses": 55000, "emi_paid": 15000},
            {"income": 102000, "expenses": 58000, "emi_paid": 16000},
            {"income": 103000, "expenses": 62000, "emi_paid": 17000},
            {"income": 103500, "expenses": 65000, "emi_paid": 18000},
            {"income": 104000, "expenses": 69000, "emi_paid": 18500},
            {"income": 104500, "expenses": 72000, "emi_paid": 19000},
        ]
    }

    response = client.post("/predict-risk", json=sequence_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_score"] == 0.82
    assert payload["risk_level"] == "HIGH"
    assert len(payload["explanations"]) == 1
