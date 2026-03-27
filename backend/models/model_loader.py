"""Load and cache trained artifacts used by inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List

import joblib

from config.settings import get_settings


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
    scaler: Any
    feature_columns: List[str]


@lru_cache(maxsize=1)
def load_artifacts() -> ModelArtifacts:
    """Load model artifacts from disk and keep them cached in memory."""

    settings = get_settings()

    classifier = joblib.load(settings.classification_model_path)
    regressor = joblib.load(settings.regression_model_path)
    scaler = joblib.load(settings.scaler_path)
    feature_columns = joblib.load(settings.feature_columns_path)

    cleaned_columns = [
        column for column in feature_columns if column not in BAD_FEATURE_COLUMNS
    ]

    return ModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        scaler=scaler,
        feature_columns=cleaned_columns,
    )
