import torch.nn as nn

class FiLMSpatial(nn.Module):

    def __init__(self,spatial_dim,text_dim):

        super().__init__()
        self.gamma = nn.Linear(text_dim, spatial_dim)
        nn.init.xavier_uniform_(self.gamma.weight)  # Initialize gamma weights with Xavier uniform
        self.beta = nn.Linear(text_dim, spatial_dim)
        nn.init.xavier_uniform_(self.beta.weight)  # Initialize beta weights with Xavier uniform

    def forward(self, feature, text):

        b, c, d, h, w = feature.shape
        
        # Generate spatial gamma and beta
        gamma = self.gamma(text).view(b, 1, d, h, w)  # shape(B, 1, D, H, W)
        beta = self.beta(text).view(b, 1, d, h, w)  # shape(B, 1, D, H, W)

        modulated = (gamma * feature) + beta
        return modulated

