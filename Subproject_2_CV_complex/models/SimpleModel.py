import torch.nn as nn
from .basemodel import BaseModel
from model_factory import register_model

@register_model
class SimpleModel(BaseModel):
    
    def __init__(self):
        super().__init__()

        def conv_block(in_features, out_features, kernel_size=(2,2), pool_size=(2,2)):
            return nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=pool_size)
            )
            
        self.model = nn.Sequential(
            conv_block(1, 1, kernel_size=5, pool_size=2),
            conv_block(1, 1, kernel_size=2, pool_size=2), 
            nn.Flatten(),
            nn.Linear(4, 1),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, x):
        return self.model(x)
    