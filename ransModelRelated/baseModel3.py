import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from ransUtils import *

class SingleInputNN(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=256, output_dim=6):
        super(SingleInputNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        x = self.dropout(self.relu(self.input_layer(inputs)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.fc4(x)

xlim = (-0.8, 1.3)
ylim = (-0.4, 0.4)

def train_single_file(file_path, epochs=1000, lr=0.001):
    data = load_single_file(file_path)
    x, y, coeffsU1, coeffsL1, coeffsU2, coeffsL2 = data["x"], data["y"], data["coeffsU1"], data["coeffsL1"], data["coeffsU2"], data["coeffsL2"]
    p, Ux, Uy, Rx, Ry, Rxy, viscosity = data["p"], data["Ux"], data["Uy"], data["Rx"], data["Ry"], data["Rxy"], data["viscosity"]
    inputs, targets = data["inputs"], data["targets"]
    scaler_input, scaler_output = data["scaler_input"], data["scaler_output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleInputNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    inputs, targets = inputs.to(device), targets.to(device)

    plot_pressure(x, y, p, 0, 'Initial Pressure Distribution', scaler_output, 'baseModel3_pressure_plot')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)

        pred_p, pred_Ux, pred_Uy, pred_Rx, pred_Ry, pred_Rxy = outputs.chunk(6, dim=1)
        loss = custom_loss(outputs, targets, x, y, pred_Ux, pred_Uy, viscosity, pred_Rx, pred_Ry, pred_Rxy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
            plot_pressure(x, y, pred_p, epoch, f'{file_path} Predicted Pressure Distribution at Epoch {epoch}', scaler_output, 'baseModel3_pressure_plot')

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_single_input_model.pth')

    print(f"Training completed. Best loss: {best_loss}")
    torch.save(model.state_dict(), 'final_single_input_model.pth')

# Get the current script's directory
base_path = Path(__file__).resolve().parent
# Training data directory name
input_data_path = 'Training_Data'
# Define relative path components for 1 csv file
csv_file = "Naca2412/Naca2412_iteration500.csv"

# Construct the full path dynamically
airfoil_file_name = base_path.parent / input_data_path / csv_file
train_single_file(airfoil_file_name)

# $ python3 ransModelRelated/baseModel3.py
# ===========================================
# Epoch 0: Loss = 348.3558654785156
# Epoch 100: Loss = 1.0033187866210938
# Epoch 200: Loss = 1.0017200708389282
# Epoch 300: Loss = 1.0008081197738647
# Epoch 400: Loss = 1.0004459619522095
# Epoch 500: Loss = 1.000264286994934
# Epoch 600: Loss = 1.0001661777496338
# Epoch 700: Loss = 1.0001946687698364
# Epoch 800: Loss = 1.0001779794692993
# Epoch 900: Loss = 1.0001922845840454
# Training completed. Best loss: 1.0001243352890015

# ===========================================
# Epoch 0: Loss = 412.0862121582031
# Epoch 100: Loss = 1.0031672716140747
# Epoch 200: Loss = 1.001726746559143
# Epoch 300: Loss = 1.0010097026824951
# Epoch 400: Loss = 1.0005295276641846
# Epoch 500: Loss = 1.0002528429031372
# Epoch 600: Loss = 1.0001806020736694
# Epoch 700: Loss = 1.0001940727233887
# Epoch 800: Loss = 1.0001624822616577
# Epoch 900: Loss = 1.0001814365386963
# Training completed. Best loss: 1.0001351833343506

# ===========================================
# Epoch 0: Loss = 858.2400512695312
# Epoch 100: Loss = 1.0034339427947998
# Epoch 200: Loss = 1.0018200874328613
# Epoch 300: Loss = 1.0009150505065918
# Epoch 400: Loss = 1.0007826089859009
# Epoch 500: Loss = 1.0005921125411987
# Epoch 600: Loss = 1.0005756616592407
# Epoch 700: Loss = 1.0005289316177368
# Epoch 800: Loss = 1.0005466938018799
# Epoch 900: Loss = 1.0005358457565308
# Training completed. Best loss: 1.0004005432128906