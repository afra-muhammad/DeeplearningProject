import torch.nn as nn
from .basemodel import BaseModel
from model_factory import register_model

@register_model
class BinaryIDSModel(BaseModel):
    """
    Binary model to classify the packets as malicious or not
    """
    def __init__(self):
        super().__init__()


        def conv_block(in_features, out_features, kernel_size=(2,2), pool_size=(2,2)):
            """Base function to create a convolution block

            Parameters
            ----------
            in_features : 
                nummber of features in
            out_features : 
                number of features out
            kernel_size : tuple, optional
                kernel size_description_, by default (2,2)
            pool_size : tuple, optional
                pool size, by default (2,2)

            Returns
            -------
            nn.Sequential
                Full convolution layer
            """
            return nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size)
            )
        
        # We create the model
        self.model = nn.Sequential(
            # Apply a first convolution layer
            conv_block(1, 16, kernel_size=5, pool_size=2),
            # Apply a second convolution layer
            conv_block(16, 32, kernel_size=3, pool_size=2),
            
            nn.Flatten(),
            # Add a dropout rate for stability
            nn.Dropout(0.2),
            # Added intermediate layer
            nn.Linear(32*2*2, 64),
            # Add some non linearity
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        """Function to initialize the weights of a model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
    