"""
Project : Advection Predictor

Akshay

"""
import torch
from torch import nn as nn
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Create initial condition

grid_points = 1000
domain = 100
grid_map = torch.linspace(0, domain, grid_points).reshape(-1, 1)
solutions = np.zeros(grid_points)
solutions[100:120] = 1

velocity = 1.0
time_step_size = 0.09
space_step = domain / (grid_points - 1)
num_time_steps = 50

# Plots the initial condition
def plot_initial_condition():
  plt.plot(grid_map, solutions)
  plt.grid(True)
  plt.xlabel('x')
  plt.ylabel('solution')
  plt.title('Initial Condition')
  plt.show()


# Generate Training data
def generate_training_data(grid_points, domain, velocity, time_step_size, space_step, solutions):
    final_solution = solutions.copy()
    CFL = velocity * (time_step_size / space_step)
    for _ in range(num_time_steps):
        f_solution = final_solution.copy()
        for i in range(1, grid_points):
            final_solution[i] = f_solution[i] - CFL * (f_solution[i] - f_solution[i - 1])

    plt.plot(grid_map, solutions)
    plt.plot(grid_map, final_solution)
    plt.grid(True)
    plt.show()
    return torch.tensor(final_solution, dtype=torch.float32).reshape(-1, 1)

    # Prepare data


y_train = generate_training_data(grid_points, domain, velocity, time_step_size, space_step, solutions)
x_train = grid_map
epochs = 10000

# Define neural network model
class AdvectionPredictor(nn.Module):
    def __init__(self):
        super(AdvectionPredictor, self).__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)  # Added another layer
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
      #print(self.layer1(x))
      x = torch.relu(self.layer1(x))
      x = torch.relu(self.layer2(x))
      x = torch.relu(self.layer3(x))  # Use ReLU activation for intermediate layers
      x = self.layer4(x)
      return x


# Initialize model, loss, and optimizer
model = AdvectionPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
def train_model(model, criterion, optimizer, x_train, y_train, epoch):
    model.train()
    train_loss = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    print(type(loss), type(loss.item()))
    print(f'First: {range(epochs)[0]} Loss : {train_loss[0]}')
    print(f'Last : {range(epochs)[-1]} Loss : {train_loss[-1]}')

    return train_loss

# Train the model
train_model(model, criterion, optimizer, x_train, y_train, epochs)

# Plot initial condition and predicted final condition
with torch.no_grad():
    model.eval()
    y_prediction = model(x_train)

plt.plot(grid_map, solutions, label='Initial Condition')
plt.plot(grid_map, y_prediction, label='Predicted Final Condition', linestyle='--', color='orange')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('solution')
plt.title('Initial Condition and Predicted Final Condition')
plt.legend()
plt.show()

