"""Baselines package exporting comparative sequence models (LSTM and CNN)."""

from hdlnn.baselines.cnn_baseline import CNNBaseline
from hdlnn.baselines.lstm_baseline import LSTMBaseline

__all__ = [
    "LSTMBaseline",
    "CNNBaseline",
]
