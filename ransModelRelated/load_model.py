import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
from ransModel import *

# Define the RANS model architecture
class ransModel(nn.Module):
    def __init__(self):
        super(ransModel, self).__init__()
        self.layer1 = nn.Linear(2, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer5 = nn.Linear(100, 100)
        self.layer6 = nn.Linear(100, 6)

    def forward(self, x, y):
        inputs = torch.cat((x, y), dim=1)
        x = torch.relu(self.layer1(inputs))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x

# Initialize the model and optimizer
ransModel = ransModel()
optimizer = optim.AdamW(ransModel.parameters(), lr=0.01)

# Load the saved state dictionary

PATH = 'ransModel.pth'
checkpoint = torch.load(PATH)
ransModel.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

ransModel.train()
runModel(f"{input('airfoil: ')}")