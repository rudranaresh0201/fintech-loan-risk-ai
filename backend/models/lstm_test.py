import torch
from lstm_model import LSTMModel

# Create model
model = LSTMModel(input_size=3, hidden_size=32)

# Dummy input (1 user, 6 months, 3 features)
# shape = (batch, seq_len, features)
dummy_input = torch.tensor([
    [
        [50000, 30000, 1],
        [52000, 31000, 1],
        [48000, 35000, 0],
        [51000, 32000, 1],
        [53000, 30000, 1],
        [49000, 34000, 0],
    ]
], dtype=torch.float32)

# Run model
output = model(dummy_input)

print("Output:", output)