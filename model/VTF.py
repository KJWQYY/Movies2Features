import torch
import torch.nn as nn

class VectorTransformer(nn.Module):
    def __init__(self, n, m):
        super(VectorTransformer, self).__init__()
        self.linear = nn.Linear(n, m)
        
    def forward(self, x):
        transformed = self.linear(x)
        return transformed
