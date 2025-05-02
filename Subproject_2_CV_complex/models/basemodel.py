import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models with common functionality"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError