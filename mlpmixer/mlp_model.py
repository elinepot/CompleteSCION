import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=32*32*3, output_dim=10, depth=3, width=64):
        super().__init__()
        self.depth = depth
        self.width = width

        # Named layers
        self.embedding = nn.Linear(input_dim, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.unembedding = nn.Linear(width, output_dim)

    def initialize_weights(self):
        nn.init.orthogonal_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

        for layer in self.hidden:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # last layer initialized with zeros
        nn.init.zeros_(self.unembedding.weight)
        nn.init.zeros_(self.unembedding.bias)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.embedding(x))
        for layer in self.hidden:
            x = torch.relu(layer(x))
        x = self.unembedding(x)
        return torch.softmax(x, dim=1)

    

if __name__=='__main__':
    import numpy as np
    mlp = MLP(depth=5, width=32)
    print(mlp)
    model_parameters = filter(lambda p: p.requires_grad, mlp.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters :', nb_params)

