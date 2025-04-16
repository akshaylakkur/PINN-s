"Rans Model with Pinn terms"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.interpolate import griddata
import ast
from ransFunction import *


# Data setup
def runModel():

    FilesNum = [2412, 2912, 4412, 4612, 4912, 5412, 5612, 5912, 6412, 6612, 6912, 7412, 7612, 7912, 8612, 8908, 9405, 9612]
    for fileNum in FilesNum:
        print(f'Starting to train Airfoil {fileNum}')
        ransData = pd.read_csv(f'/home/cde1/ml/akshay/Training_Data/Naca{str(fileNum)}/Naca{str(fileNum)}_iteration500.csv', quotechar='"')
        ransData.columns = ransData.columns.str.strip()
        ransData = pd.DataFrame(ransData)


        # Convert to tensors/setup training
        x = torch.tensor(ransData['x'].to_list()).unsqueeze(1).float().requires_grad_(True)
        y = torch.tensor(ransData['y'].to_list()).unsqueeze(1).float().requires_grad_(True)
        p = torch.tensor(ransData['p'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Ux = torch.tensor(ransData['Ux'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Uy = torch.tensor(ransData['Uy'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Rx = torch.tensor(ransData['Rx'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Ry = torch.tensor(ransData['viscosity(nut)'].to_list()).unsqueeze(1).float().requires_grad_(True)
        Rxy = torch.tensor(ransData['Ry'].to_list()).unsqueeze(1).float().requires_grad_(True)
        viscosity = torch.tensor(ransData['Rxy'].to_list()).unsqueeze(1).float().requires_grad_(True)
        coeffsU1 = torch.tensor(ransData['coeffsU1'].apply(ast.literal_eval)).float().requires_grad_(True)
        coeffsL1 = torch.tensor(ransData['coeffsL1'].apply(ast.literal_eval)).float().requires_grad_(True)
        coeffsU2 = torch.tensor(ransData['coeffsU2'].apply(ast.literal_eval)).float().requires_grad_(True)
        coeffsL2 = torch.tensor(ransData['coeffsL2'].apply(ast.literal_eval)).float().requires_grad_(True)

        coeffs = torch.cat([coeffsU1, coeffsL1, coeffsU2, coeffsL2], dim = 1)

        # NN architecture
        # class ransModel(nn.Module):
        #     def __init__(self):
        #         super(ransModel, self).__init__()
        #         self.layer1 = nn.Linear(34, 100)
        #         self.layer2 = nn.Linear(100, 100)
        #         self.layer5 = nn.Linear(100, 100)
        #         self.layer6 = nn.Linear(100, 6)
        #         self.coefficients = None
        #
        #     'set airfoil coordinates once'
        #     def set_coordinates(self, coeffsU1, coeffsL1, coeffsU2, coeffsL2):
        #         self.coefficients = torch.cat([coeffsU1, coeffsL1, coeffsU2, coeffsL2], dim=1)
        #
        #     def forward(self, x, y):
        #         expandedCoeffs = self.coefficients.expand(x.shape[0], -1)
        #         inputs = torch.cat((x, y, expandedCoeffs), dim=1)
        #         x = torch.relu(self.layer1(inputs))
        #         x = torch.relu(self.layer2(x))
        #         x = torch.relu(self.layer5(x))
        #         x = self.layer6(x)
        #         return x

        class ransModel(nn.Module):
            def __init__(self):
                super(ransModel, self).__init__()
                self.coeffEncoder = nn.Sequential(
                    nn.Linear(32, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256)
                )

                self.coordsEncoder = nn.Sequential(
                    nn.Linear(2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256)
                )
                self.combine = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 6)
                )

            def forward(self, coeffs, coords):
                coeffFeatures = self.coeffEncoder(coeffs)
                coeffFeatures = coeffFeatures.unsqueeze(1).expand(-1, 1, coords.shape[1], -1)
                
                coordinates = coords.view(-1, 2)
                coordFeatures = self.coordsEncoder(coordinates)
                coordFeatures = coordFeatures.view(coords.shape[0], coords.shape[1], -1)
                
                combinedFeatures = torch.cat([coeffFeatures.squeeze(0), coordFeatures], dim = 2)
                output = self.combine(combinedFeatures)
                return output


        # Initialize the model and optimizer
        ransModel = ransModel()
        #ransModel.set_coordinates(coeffsU1, coeffsL1, coeffsU2, coeffsL2)
        optimizer = optim.AdamW(ransModel.parameters(), lr=0.01)
        basicLoss = nn.MSELoss()


        # Training Loop
        def ransTrain(epochs):
            ransModel.train()
            coords = torch.cat([x,y], dim=1).unsqueeze(0)
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = ransModel(coeffs.unsqueeze(0), coords)

                # Splitting the outputs to match the physical variables
                pred_p, pred_Ux, pred_Uy, pred_Rx, pred_Ry, pred_Rxy = torch.split(outputs.squeeze(0), 1, dim=1)

                # MSE Loss between predicted and true values
                traditionalLoss = basicLoss(outputs.squeeze(0), torch.cat((p, Ux, Uy, Rx, Ry, Rxy), dim=1))

                # Compute residuals
                c = continuity(x, y, pred_Ux, pred_Uy)
                u_momentum = momentum_u(x, y, pred_Ux, pred_Uy, 1, pred_p, viscosity, pred_Rx, pred_Rxy)
                v_momentum = momentum_v(x, y, pred_Ux, pred_Uy, 1, pred_p, viscosity, pred_Ry, pred_Rxy)

                # Total PINN loss
                pinnLoss = c + u_momentum + v_momentum

                # Combine losses
                ransLoss = traditionalLoss + 0.01 * pinnLoss

                # Backpropagation and optimization
                ransLoss.backward()
                optimizer.step()

                if epoch % 250 == 0:
                    print(f'Epoch {epoch} - Total Loss: {ransLoss.item()}, Continuity Loss: {c.item()}, U Momentum Loss: {u_momentum.item()}, V Momentum Loss: {v_momentum.item()}, MSE Loss: {traditionalLoss.item()}')

        # Start training
        ransTrain(3001)



        # Save the model state dictionary
        PATH = 'ransModel.pth'
        torch.save({
            'model_state_dict': ransModel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)


        print(f"Model saved successfully for airfoil {fileNum}.")    
runModel()
