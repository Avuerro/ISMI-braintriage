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
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_in, n_features)  
        self.fc2 = nn.Linear(n_features,1)
        #self.sm1 = nn.Softmax(1)

    def forward(self, x):
        print(f'Start resnet {x.shape}')
        x = self.model(x)
        print(f'After resnet layers {x.shape}')
        # Turn x into the right shape
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)
        print(f'After reshaping {x.shape}')
        # Put output x through our self defined layers
        x = self.fc1(x)
        print(f'After first FC layer {x.shape}')
        out = self.fc2(x)
        print(f'After final FC layer {out.shape}')
        #print(out)

        
        return out.view(-1)