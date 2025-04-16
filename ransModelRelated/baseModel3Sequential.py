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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the models sequentially
# The first model is trained on first data, then it is used to train on the second data, for all data
def train_on_file(model, file_name, optimizer, epochs=1000, lr=0.01):
    data = load_single_file(file_name)
    x, y, coeffsU1, coeffsL1, coeffsU2, coeffsL2 = data["x"], data["y"], data["coeffsU1"], data["coeffsL1"], data["coeffsU2"], data["coeffsL2"]
    p, Ux, Uy, Rx, Ry, Rxy, viscosity = data["p"], data["Ux"], data["Uy"], data["Rx"], data["Ry"], data["Rxy"], data["viscosity"]
    inputs, targets = data["inputs"], data["targets"]
    scaler_input, scaler_output = data["scaler_input"], data["scaler_output"]

    model.train()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    inputs, targets = inputs.to(device), targets.to(device)

    plot_pressure(x, y, p, 0, 'Initial Pressure Distribution', scaler_output, 'sequential_baseModel3_pressure_plot')

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
            plot_pressure(x, y, pred_p, epoch, f"{file_name} Predicted Pressure Distribution at Epoch {epoch}", scaler_output, 'sequential_baseModel3_pressure_plot')

        if loss.item() < best_loss:
            best_loss = loss.item()

    print(f"{folder} Best loss: {best_loss}")
    plot_pressure(x, y, p, None, f'Actual Pressure - File {file_name.split("/")[-2]}', scaler_output, pngFilePrefix='Actual')
    plot_pressure(x, y, pred_p, None,f'Final Predicted Pressure - File {file_name.split("/")[-2]}', scaler_output,'final_sequential_baseModel3_pressure_plot')

    return model

input_data_path = 'Training_Data'
airfoil_file_pattern = "_iteration500.csv"
current_dir = Path(__file__).parent
training_data_path = current_dir.parent / input_data_path
all_data = get_csv_files(training_data_path, airfoil_file_pattern)

epochs = 200
outputModel = 'sequentialModels_final_baseModel3_model.pth'
lr = 0.01

# Initialize the model
model = SingleInputNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

for i, file_name in enumerate(all_data):
    folder = file_name.split("/")[-2]
    print(f"Training on file {folder}")

    checkpoint_path = f'sequentialModels_baseModel3_model_state_after_file_{i-1}.pth'
    if i > 0 and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Allow minor mismatches
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_keys = set(torch.load('PINNmodel.pth').keys())
        current_keys = set(model.state_dict().keys())

        if saved_keys != current_keys:
            print("WARNING: Model layer name mismatch detected!")
            print("Missing in current model:", saved_keys - current_keys)
            print("Unexpected in checkpoint:", current_keys - saved_keys)

    # Train the model
    model = train_on_file(model, file_name, optimizer, epochs, lr=lr)

    # Save the model state after training
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'sequentialModels_baseModel3_model_state_after_file_{i}.pth')

print("Sequential training completed.")

# Save the final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, outputModel)

print(f"Final sequential model {outputModel} saved successfully.")

#  % python3 ransModelRelated/baseModel3Sequential.py
# Training on file Naca8908
# Epoch 0: Loss = 3633.024169921875
# Epoch 100: Loss = 37.05946350097656
# Naca8908 Best loss: 3.005042314529419
# Training on file Naca7912
# Epoch 0: Loss = 30343.85546875
# Epoch 100: Loss = 1.3491666316986084
# Naca7912 Best loss: 1.017622709274292
# Training on file Naca5612
# Epoch 0: Loss = 50762.39453125
# Epoch 100: Loss = 1.0030791759490967
# Naca5612 Best loss: 1.0001552104949951
# Training on file Naca6912
# Epoch 0: Loss = 51.826568603515625
# Epoch 100: Loss = 1.090009331703186
# Naca6912 Best loss: 1.000877022743225
# Training on file Naca4612
# Epoch 0: Loss = 923.2145385742188
# Epoch 100: Loss = 1.0001766681671143
# Naca4612 Best loss: 1.0000404119491577
# Training on file Naca2912
# Epoch 0: Loss = 1.0021429061889648
# Epoch 100: Loss = 1.000158667564392
# Naca2912 Best loss: 0.9999679923057556
# Training on file Naca7412
# Epoch 0: Loss = 1.0971097946166992
# Epoch 100: Loss = 1.0001219511032104
# Naca7412 Best loss: 0.9999478459358215
# Training on file Naca2412
# Epoch 0: Loss = 1.000035047531128
# Epoch 100: Loss = 1.0000075101852417
# Naca2412 Best loss: 0.9999483227729797
# Training on file Naca6412
# Epoch 0: Loss = 0.9999810457229614
# Epoch 100: Loss = 0.9999699592590332
# Naca6412 Best loss: 0.9999374151229858
# Training on file Naca5412
# Epoch 0: Loss = 1.0000208616256714
# Epoch 100: Loss = 1.000052809715271
# Naca5412 Best loss: 0.9999437928199768
# Training on file Naca9612
# Epoch 0: Loss = 0.9999745488166809
# Epoch 100: Loss = 1.0000081062316895
# Naca9612 Best loss: 0.9999366998672485
# Training on file Naca4412
# Epoch 0: Loss = 1.0000030994415283
# Epoch 100: Loss = 0.9999895095825195
# Naca4412 Best loss: 0.9999198317527771
# Training on file Naca8612
# Epoch 0: Loss = 1.0000146627426147
# Epoch 100: Loss = 0.9999798536300659
# Naca8612 Best loss: 0.9998825192451477
# Training on file Naca5912
# Epoch 0: Loss = 1.000182867050171
# Epoch 100: Loss = 1.000138759613037
# Naca5912 Best loss: 0.9998375773429871
# Training on file Naca7612
# Epoch 0: Loss = 1.0000230073928833
# Epoch 100: Loss = 1.0000245571136475
# Naca7612 Best loss: 0.9999315738677979
# Training on file Naca9405
# Epoch 0: Loss = 1.000006079673767
# Epoch 100: Loss = 1.0000147819519043
# Naca9405 Best loss: 0.9999735355377197
# Training on file Naca4912
# Epoch 0: Loss = 1.0000971555709839
# Epoch 100: Loss = 1.000016212463379
# Naca4912 Best loss: 0.9998791217803955
# Training on file Naca6612
# Epoch 0: Loss = 1.0000401735305786
# Epoch 100: Loss = 1.0000368356704712
# Naca6612 Best loss: 0.9999293088912964
# Sequential training completed.
# Final sequential model sequentialModels_final_baseModel3_model.pth saved successfully.