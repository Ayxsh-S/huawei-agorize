import torch
import torch.nn as nn

class EarLandmarNet(nn.Module):
    def __init__(self, in_dim: int = 3, n_landmarks: int = 85):