import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, n_features, n_hidden, is_bidirectional=True, n_layers=1):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.is_bidirectional = is_bidirectional
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, bidirectional=is_bidirectional)
        if is_bidirectional:
            self.classification_linear = nn.Linear(n_hidden * 2, 1)
        else:
            self.classification_linear = nn.Linear(n_hidden, 1)
        # For saving model and loss values
        self.name = "LSTMNet"

    def forward(self, x):
        batch_size, n_slices, _ = x.shape
        lstm_output, (_, _) = self.lstm(x)
        # Common method for LSTMs is to take the last sequence output and pass that to a dense layer
        last_sequence_output = torch.squeeze(lstm_output[:, -1]) # Shape: batch, n_hidden (after squeeze)
        output = self.classification_linear(torch.squeeze(last_sequence_output))
        return output