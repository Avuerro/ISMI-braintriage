import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, 
                            num_layers=n_layers, batch_first=True)
        self.classification_linear = nn.Linear(n_hidden,1)
        # For saving model and loss values
        self.name = "LSTMNet" 
        
    def forward(self, x):
        lstm_output, (_, _) = self.lstm(x)
        output = self.classification_linear(lstm_output[:,-1])
        return torch.squeeze(output)