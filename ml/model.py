"""Training helpers for classifier and regressor artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedArtifacts:
    """Represents fitted artifacts from one training run."""

    classifier: Any
    regressor: Any
    scaler: StandardScaler
    feature_columns: list[str]


def train_models(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series) -> TrainedArtifacts:
    """Train baseline logistic and linear models."""

    feature_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_scaled, y_class)

    regressor = LinearRegression()
    regressor.fit(X_scaled, y_reg)

    return TrainedArtifacts(
        classifier=classifier,
        regressor=regressor,
        scaler=scaler,
        feature_columns=feature_columns,
    )


def save_artifacts(artifacts: TrainedArtifacts, output_dir: Path) -> None:
    """Persist trained artifacts to the provided directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.classifier, output_dir / "classification_model.pkl")
    joblib.dump(artifacts.regressor, output_dir / "regression_model.pkl")
    joblib.dump(artifacts.scaler, output_dir / "scaler.pkl")
    joblib.dump(artifacts.feature_columns, output_dir / "feature_columns.pkl")
