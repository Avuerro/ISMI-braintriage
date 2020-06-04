import torch
import torch.nn as nn


class FeatureVectorModel(nn.Module):
    def __init__(self, n_features):
        super(FeatureVectorModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(512*512*3, 256)
        self.feature_linear = nn.Linear(256, n_features)
        self.classification_linear = nn.Linear(n_features, 1)
        # For saving model and loss values
        self.name = "FCNet" 
        
    def forward(self, x):
        h = self.linear_1(self.flatten(x))
        h = self.feature_linear(h)
        h = self.classification_linear(h)
        # We don't need Softmax when using BCEWithLogitsLoss
        return torch.squeeze(h)