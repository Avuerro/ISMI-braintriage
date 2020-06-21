import torch
import torch.nn as nn


class CombinedNet(nn.Module):
    def __init__(self, name, cnn_net, lstm_net, do_freeze_fc_net=False):
        super(CombinedNet, self).__init__()
        # Remove classification layer from FC network
        self.cnn_net = torch.nn.Sequential(*(list(cnn_net.children())[:-1]))
        self.lstm_net = lstm_net
        # For saving model and loss values
        self.name = name

    def forward(self, batch_of_patients):
        # Loop over all slices and compute feature vectors with cnn_net
        feature_vectors = [self.cnn_net(batch_of_slices) for batch_of_slices in batch_of_patients]
        # Create tensor with shape (batch, slices, features)
        feature_vectors = torch.stack(feature_vectors)

        # Compute predictions with LSTM net
        predictions = self.lstm_net(feature_vectors)

        return predictions

    def set_learning_cnn_net(self, do_learning):
        for param in self.cnn_net.parameters():
            param.requires_grad = do_learning