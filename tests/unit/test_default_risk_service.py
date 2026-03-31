"""Unit tests for LSTM default-risk service."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from backend.services import default_risk_service
from schemas.requests import RiskSequenceRequest


class DummyRiskModel(torch.nn.Module):
    """Torch model stub returning deterministic high-risk score."""

    def forward(self, sequence_batch: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.82], dtype=torch.float32)


def test_predict_default_risk_returns_score_level_and_explanations(monkeypatch) -> None:
    """Service should return normalized score and HIGH risk level for high logits."""

    def fake_load_default_risk_artifacts() -> SimpleNamespace:
        return SimpleNamespace(
            model=DummyRiskModel(),
            mean=np.array([100_000.0, 60_000.0, 20_000.0], dtype=np.float32),
            std=np.array([10_000.0, 8_000.0, 4_000.0], dtype=np.float32),
            sequence_length=6,
            feature_order=["income", "expenses", "emi_paid"],
        )

    monkeypatch.setattr(
        default_risk_service,
        "load_default_risk_artifacts",
        fake_load_default_risk_artifacts,
    )

    payload = RiskSequenceRequest(
        sequence=[
            {"income": 90_000, "expenses": 55_000, "emi_paid": 15_000},
            {"income": 91_000, "expenses": 57_000, "emi_paid": 16_000},
            {"income": 92_000, "expenses": 60_000, "emi_paid": 17_000},
            {"income": 93_000, "expenses": 64_000, "emi_paid": 18_000},
            {"income": 94_000, "expenses": 68_000, "emi_paid": 19_000},
            {"income": 95_000, "expenses": 73_000, "emi_paid": 20_000},
        ]
    )

    response = default_risk_service.predict_default_risk(payload)

    assert response.risk_score == 0.82
    assert response.risk_level == "HIGH"
    assert len(response.explanations) > 0
