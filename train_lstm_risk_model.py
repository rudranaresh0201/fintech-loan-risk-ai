"""End-to-end LSTM loan risk model with training, save/load, inference, and explanation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_ORDER = ["income", "expenses", "emi_paid"]
SEQUENCE_LENGTH = 6
N_FEATURES = 3
MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
SCALER_PATH = Path(__file__).resolve().parent / "scaler.pkl"


def generate_synthetic_dataset(
    n_samples: int = 3000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequences with shape (N, 6, 3) and binary labels.

    Label rule:
    - 1 when expenses increase across the window AND income decreases.
    - 0 otherwise.
    """

    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        start_income = float(rng.uniform(35000, 90000))
        income_trend = float(rng.uniform(-2500, 2500))
        income_noise = rng.normal(0, 1000, size=SEQUENCE_LENGTH)
        income = start_income + income_trend * np.arange(SEQUENCE_LENGTH) + income_noise
        income = np.clip(income, 15000, None)

        start_expenses = float(rng.uniform(15000, 45000))
        expense_trend = float(rng.uniform(-1800, 1800))
        expense_noise = rng.normal(0, 800, size=SEQUENCE_LENGTH)
        expenses = start_expenses + expense_trend * np.arange(SEQUENCE_LENGTH) + expense_noise
        expenses = np.clip(expenses, 3000, None)

        emi_ratio = float(rng.uniform(0.08, 0.25))
        emi_noise = rng.normal(0, 400, size=SEQUENCE_LENGTH)
        emi_paid = income * emi_ratio + emi_noise
        emi_paid = np.clip(emi_paid, 500, None)

        X[i, :, 0] = income
        X[i, :, 1] = expenses
        X[i, :, 2] = emi_paid

        expenses_increase = expenses[-1] > expenses[0]
        income_decrease = income[-1] < income[0]
        y[i] = 1.0 if (expenses_increase and income_decrease) else 0.0

    return X, y


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple deterministic train-validation split."""

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    cut = int(len(X) * (1 - val_ratio))
    train_idx = idx[:cut]
    val_idx = idx[cut:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def preprocess_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train data and transform train/val correctly."""

    scaler = StandardScaler()

    train_flat = X_train.reshape(-1, N_FEATURES)
    val_flat = X_val.reshape(-1, N_FEATURES)

    train_scaled_flat = scaler.fit_transform(train_flat)
    val_scaled_flat = scaler.transform(val_flat)

    X_train_scaled = train_scaled_flat.reshape(X_train.shape).astype(np.float32)
    X_val_scaled = val_scaled_flat.reshape(X_val.shape).astype(np.float32)

    return X_train_scaled, X_val_scaled, scaler


class LoanRiskLSTM(nn.Module):
    """LSTM binary classifier that outputs one probability per sequence."""

    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq_len, features) -> output shape: (batch,)."""

        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        probs = self.sigmoid(logits)
        return probs.squeeze(1)


def train_model(
    model: LoanRiskLSTM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> LoanRiskLSTM:
    """Train model using DataLoader, BCELoss, and Adam on CPU."""

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += float(loss.item())

        avg_train = total_train_loss / max(len(train_loader), 1)
        avg_val = total_val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

    return model


def save_artifacts(
    model: LoanRiskLSTM,
    scaler: StandardScaler,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> None:
    """Save model.pt and scaler.pkl artifacts."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": N_FEATURES,
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_order": FEATURE_ORDER,
    }
    torch.save(checkpoint, model_path)
    joblib.dump(scaler, scaler_path)


def load_artifacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> tuple[LoanRiskLSTM, StandardScaler]:
    """Load trained model and fitted scaler."""

    checkpoint = torch.load(model_path, map_location="cpu")
    model = LoanRiskLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler: StandardScaler = joblib.load(scaler_path)
    return model, scaler


def _validate_input_sequence(sequence: np.ndarray) -> np.ndarray:
    """Ensure inference input is a single sequence with shape (6, 3)."""

    arr = np.asarray(sequence, dtype=np.float32)
    if arr.shape != (SEQUENCE_LENGTH, N_FEATURES):
        raise ValueError(
            f"Expected sequence shape ({SEQUENCE_LENGTH}, {N_FEATURES}), got {arr.shape}."
        )
    return arr


def predict_probability(
    model: LoanRiskLSTM,
    scaler: StandardScaler,
    sequence: np.ndarray,
) -> float:
    """Return default-risk probability for one sequence."""

    arr = _validate_input_sequence(sequence)
    scaled = scaler.transform(arr).astype(np.float32)
    batch = torch.tensor(scaled.reshape(1, SEQUENCE_LENGTH, N_FEATURES), dtype=torch.float32)

    with torch.no_grad():
        prob = float(model(batch).item())

    return prob


def explain_prediction(sequence: np.ndarray, probability: float) -> str:
    """Provide a simple human-readable explanation from sequence trends."""

    arr = _validate_input_sequence(sequence)
    income_start, income_end = float(arr[0, 0]), float(arr[-1, 0])
    expense_start, expense_end = float(arr[0, 1]), float(arr[-1, 1])

    income_change = income_end - income_start
    expense_change = expense_end - expense_start

    income_direction = "decreased" if income_change < 0 else "increased"
    expense_direction = "increased" if expense_change > 0 else "decreased"

    return (
        f"Predicted risk probability is {probability:.3f}. "
        f"Income {income_direction} by {abs(income_change):.2f}, "
        f"while expenses {expense_direction} by {abs(expense_change):.2f}."
    )


def main() -> None:
    # 1) Create synthetic data
    X, y = generate_synthetic_dataset(n_samples=3000, seed=42)
    print("Dataset shape:", X.shape, "Labels shape:", y.shape)

    # 2) Split
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)

    # 3) Preprocess with StandardScaler
    X_train_scaled, X_val_scaled, scaler = preprocess_sequences(X_train, X_val)

    # 4) Train LSTM
    model = LoanRiskLSTM(input_size=N_FEATURES, hidden_size=32, num_layers=1)
    model = train_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        epochs=12,
        batch_size=64,
        lr=1e-3,
    )

    # 5) Save artifacts
    save_artifacts(model=model, scaler=scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")

    # 6) Load artifacts
    loaded_model, loaded_scaler = load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

    # 7) Inference on one validation sequence
    test_sequence = X_val[0]
    probability = predict_probability(loaded_model, loaded_scaler, test_sequence)
    explanation = explain_prediction(test_sequence, probability)

    print(f"Test probability: {probability:.4f}")
    print("Explanation:", explanation)


if __name__ == "__main__":
    main()
