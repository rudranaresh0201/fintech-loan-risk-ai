"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    project_name: str = "EMI Predict AI API"
    project_version: str = "1.0.0"
    model_dir: Path = BASE_DIR
    classification_model_file: str = os.getenv(
        "CLASSIFICATION_MODEL_FILE", "classification_model.pkl"
    )
    regression_model_file: str = os.getenv(
        "REGRESSION_MODEL_FILE", "regression_model.pkl"
    )
    risk_model_file: str = os.getenv("RISK_MODEL_FILE", "model.pkl")
    default_risk_lstm_file: str = os.getenv(
        "DEFAULT_RISK_LSTM_FILE", "default_risk_lstm.pt"
    )
    default_risk_normalizer_file: str = os.getenv(
        "DEFAULT_RISK_NORMALIZER_FILE", "default_risk_normalizer.pkl"
    )
    scaler_file: str = os.getenv("SCALER_FILE", "scaler.pkl")
    feature_columns_file: str = os.getenv("FEATURE_COLUMNS_FILE", "feature_columns.pkl")

    @property
    def classification_model_path(self) -> Path:
        return self.model_dir / self.classification_model_file

    @property
    def regression_model_path(self) -> Path:
        return self.model_dir / self.regression_model_file

    @property
    def risk_model_path(self) -> Path:
        return self.model_dir / self.risk_model_file

    @property
    def default_risk_lstm_path(self) -> Path:
        return self.model_dir / self.default_risk_lstm_file

    @property
    def default_risk_normalizer_path(self) -> Path:
        return self.model_dir / self.default_risk_normalizer_file

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / self.scaler_file

    @property
    def feature_columns_path(self) -> Path:
        return self.model_dir / self.feature_columns_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""

    return Settings()
