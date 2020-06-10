import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, bidirectional=True)
        self.classification_linear = nn.Linear(n_hidden, 1)
        # For saving model and loss values
        self.name = "LSTMNet"

    def forward(self, x):
        lstm_output, (_, _) = self.lstm(x)
        if (lstm_output.shape[2] == 2 * n_hidden):  # This way it works for both bidirectional and a unidirectional LSTM
            batch_size = lstm_output.shape[0]
            lstm_output = lstm_output.view(batch_size, 2, n_hidden)  # batch size, num_directions, n_hidden)
            lstm_output = lstm_output[:, 0, :].view(batch_size, 1, n_hidden)
            # output of shape (seq_len, batch, num_directions * hidden_size):
            # For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size),
            # with forward and backward being direction 0 and 1 respectively.
            # Similarly, the directions can be separated in the packed case.
        output = self.classification_linear(lstm_output[:, -1])
        return torch.squeeze(output)