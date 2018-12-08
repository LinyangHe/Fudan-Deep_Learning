import torch.nn as nn
import torch

class Pointer_network(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = nn.LSTm
        self.decoder = 
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

