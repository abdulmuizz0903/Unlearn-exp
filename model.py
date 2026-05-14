import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256], num_classes=10):
        super(SimpleMLP, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
            
        layers = []
        in_size = input_size
        for h in hidden_sizes[:-1]:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
            
        self.feature_extractor = nn.Sequential(*layers) if len(hidden_sizes) > 1 else nn.Identity()
        self.targeted_layer = nn.Linear(in_size, hidden_sizes[-1])
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        self.hidden_activations = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature_extractor(x)
        x = self.targeted_layer(x)
        x = F.relu(x)
        self.hidden_activations = x
        x = self.output_layer(x)
        return x
