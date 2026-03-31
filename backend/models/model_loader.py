"""Load and cache trained artifacts used by inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List

import joblib
import numpy as np
import torch

from config.settings import get_settings
from ml.lstm_default_risk_model import LSTMDefaultRiskModel


BAD_FEATURE_COLUMNS = {
    "emi_eligibility_High_Risk",
    "emi_eligibility_Not_Eligible",
    "emi_eligibility_label",
}


@dataclass(frozen=True)
class ModelArtifacts:
    """Container for all loaded model artifacts."""

    classifier: Any
    regressor: Any
    risk_model: Any
    scaler: Any
    feature_columns: List[str]


@dataclass(frozen=True)
class DefaultRiskArtifacts:
    """Container for LSTM-based default risk inference artifacts."""

    model: LSTMDefaultRiskModel
    mean: np.ndarray
    std: np.ndarray
    sequence_length: int
    feature_order: List[str]


@lru_cache(maxsize=1)
def load_artifacts() -> ModelArtifacts:
    """Load model artifacts from disk and keep them cached in memory."""

    settings = get_settings()

    classifier = joblib.load(settings.classification_model_path)
    regressor = joblib.load(settings.regression_model_path)
    risk_model = (
        joblib.load(settings.risk_model_path)
        if settings.risk_model_path.exists()
        else None
    )
    scaler = joblib.load(settings.scaler_path)
    feature_columns = joblib.load(settings.feature_columns_path)

    cleaned_columns = [
        column for column in feature_columns if column not in BAD_FEATURE_COLUMNS
    ]

    return ModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        risk_model=risk_model,
        scaler=scaler,
        feature_columns=cleaned_columns,
    )


@lru_cache(maxsize=1)
def load_default_risk_artifacts() -> DefaultRiskArtifacts | None:
    """Load and cache LSTM default-risk artifacts when available."""

    settings = get_settings()

    if not settings.default_risk_lstm_path.exists():
        return None

    if not settings.default_risk_normalizer_path.exists():
        return None

    checkpoint = torch.load(settings.default_risk_lstm_path, map_location="cpu")
    normalizer = joblib.load(settings.default_risk_normalizer_path)

    input_size = int(checkpoint.get("input_size", 3))
    hidden_size = int(checkpoint.get("hidden_size", 64))
    num_layers = int(checkpoint.get("num_layers", 2))
    dropout = float(checkpoint.get("dropout", 0.2))

    model = LSTMDefaultRiskModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    sequence_length = int(normalizer.get("sequence_length", 6))
    feature_order = list(normalizer.get("feature_order", ["income", "expenses", "emi_paid"]))

    return DefaultRiskArtifacts(
        model=model,
        mean=mean,
        std=std,
        sequence_length=sequence_length,
        feature_order=feature_order,
    )
