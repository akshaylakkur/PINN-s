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

def train_single_file(file_path, outputModel, epochs=1000, lr=0.001):
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

    plot_pressure(x, y, p, 0, 'Initial Pressure Distribution', scaler_output, 'initial')

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

        folder = file_path.split("/")[-2]
        if epoch % 100 == 0:
            print(f"{folder} Epoch {epoch}: Loss = {loss.item()}")
            plot_pressure(x, y, pred_p, epoch, f'{file_path} Predicted Pressure Distribution at Epoch {epoch}', scaler_output, 'separate_pressure_plot')

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"separated_{folder}_best_single_input_model.pth")

    print(f"Training {folder} completed. Best loss: {best_loss}")
    torch.save(model.state_dict(), outputModel)

input_data_path = 'Training_Data'
airfoil_file_pattern = "_iteration500.csv"

# Get the directory of the current script
current_dir = Path(__file__).parent

# Get the parent directory and construct the path to training_data
training_data_path = current_dir.parent / input_data_path

# Combine the *_iteration500.csv into single datasets from all sub-folders
all_data = get_csv_files(training_data_path, airfoil_file_pattern)

models = []
# Train all input files separately and get models separately
for data in all_data:
    folderName = data.split("/")[-2]
    print(f'Training on file {folderName}')
    outputModel = f"separateModels_{folderName}_baseModel3.pth"
    train_single_file(data, outputModel)
    models.append(outputModel)

print(models)

# $ python3 ransModelRelated/baseModel3SeparateModels.py
# Training on file Naca8908
# Naca8908 Epoch 0: Loss = 6271.59423828125
# Naca8908 Epoch 100: Loss = 1.0035922527313232
# Naca8908 Epoch 200: Loss = 1.0007524490356445
# Naca8908 Epoch 300: Loss = 1.000197172164917
# Naca8908 Epoch 400: Loss = 1.000349521636963
# Naca8908 Epoch 500: Loss = 1.0002802610397339
# Naca8908 Epoch 600: Loss =  1.0002572536468506
# Naca8908 Epoch 700: Loss = 1.0002658367156982