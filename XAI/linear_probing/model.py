import torch
import torch.nn as nn

class LinearProbeModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(LinearProbeModel, self).__init__()

        self.encoder = encoder
        self.linear_layer = nn.Linear(self.encoder.OUTPUT_DIM, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        features = features.to(torch.float32)
        features = torch.flatten(features, start_dim=1)
        output = self.linear_layer(features)

        return output