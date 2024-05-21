import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from .model import Autoencoder
from .utils import (
    LossFunctionType,
    create_sequences,
    get_loss_function,
    set_seeds,
)


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def detect_anomalies_with_lstm_autoencoder(part_data, seq_length=10, hidden_dim=30, epochs=200, lr=1e-3, verbose=False, threshold_percentile=99, seed=42):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    part_data_scaled = scaler.fit_transform(part_data)
    part_sequences = create_sequences(part_data_scaled, seq_length)
    part_sequences_tensor = torch.FloatTensor(part_sequences)

    model = Autoencoder(part_sequences_tensor.shape[2], hidden_dim, seq_length)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(part_sequences_tensor)
        loss = criterion(outputs, part_sequences_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1:>3}/{epochs:>3}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        reconstructed = model(part_sequences_tensor)
        reconstruction_errors = torch.mean((part_sequences_tensor - reconstructed) ** 2, dim=[1, 2])

    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomaly_indices = np.where(reconstruction_errors > threshold)[0]

    return anomaly_indices, reconstruction_errors


def detect_anomalies_with_moving_avg_std(
    data: np.ndarray,
    window_size: int = 10,
    threshold_factor: float = 2.5,
):
    num_dimensions = data.shape[1]
    squared_residuals_per_dimension = []

    for dim in range(num_dimensions):
        dim_data = data[:, dim]

        moving_avg = np.convolve(
            dim_data, np.ones(window_size) / window_size, mode="valid"
        )

        squared_residuals = (
            dim_data[window_size - 1 :] - moving_avg  # noqa
        ) ** 2
        squared_residuals_per_dimension.append(squared_residuals)

    distances = np.sqrt(
        np.sum(np.array(squared_residuals_per_dimension), axis=0)
    )

    moving_avg_distance = np.convolve(
        distances, np.ones(window_size) / window_size, mode="valid"
    )

    moving_std_distance = [
        np.std(distances[i - window_size : i])  # noqa
        for i in range(window_size, len(distances) + 1)
    ]

    upper_bound = moving_avg_distance + threshold_factor * np.array(
        moving_std_distance
    )

    anomaly_indices = (
        np.where(distances[window_size - 1 :] > upper_bound)[0]  # noqa
        + window_size
        - 1
    )

    return anomaly_indices
