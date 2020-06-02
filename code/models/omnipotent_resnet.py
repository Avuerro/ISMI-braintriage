import torch 
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, model,name,n_features):
        super(Net, self).__init__()
        self.model = model
        fc_in = model.fc.in_features
        
        # Remove last layer of original model
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.name = name
        # Final layers
        self.fc1 = nn.Linear(fc_in, n_features)  
        self.fc2 = nn.Linear(128,1)
        #self.sm1 = nn.Softmax(1)

    def forward(self, x):
        x = self.model(x)
        # Turn x into the right shape
        x = x.view(x.size(0), -1)
        
        # Put output x through our self defined layers
        x = self.fc1(x)
        out = self.fc2(x)
        #print(out)

        
        return out.view(-1)