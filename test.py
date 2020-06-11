import torch.nn as nn

a=(nn.Linear(1, 2), nn.Linear(2, 2))
nn.Sequential(*a)