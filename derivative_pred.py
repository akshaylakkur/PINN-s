"""
Project 1: Derivative Predictor
"""
#Akshay
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np


# Function to calculate derivative using forward difference method
def forward_difference(func, x, h=0.000001):
    return (func(x + h) - func(x)) / h

# Function to define the original function f(x) = x^2
def function(x):
    return x**2

# Function to approximate derivatives for each x value
def func_derivative(x_values):
    derivatives = []
    for x in x_values:
        x = x.item()
        approx_derivative = forward_difference(function, x)
        derivatives.append(approx_derivative)
    return torch.tensor(derivatives).reshape(-1, 1)

# Generate data for training
def generate_data():
    x_values = torch.linspace(0, 100, 100).reshape(-1, 1)
    y_values = func_derivative(x_values)
    return x_values, y_values

# Neural network model for predicting derivatives
class DerivativePredictor(nn.Module):
    def __init__(self):
        super(DerivativePredictor, self).__init__()
        # bigger the nn, the more accurate results are but slower it takes to run
        self.layer1 = nn.Linear(1, 16)  # layer 1 of nn size
        self.layer2 = nn.Linear(16, 1)  # layer 2 of nn size
# forward pass of function
    def forward(self, x):
        x = torch.relu(self.layer1(x)) #activation function (relu)
        x = self.layer2(x)
        return x


# Generate data and prepare for training
x_train, y_train = generate_data()
model = DerivativePredictor()
#set loss (could also be L1 (check torch.losses))
losses = nn.MSELoss()
#set optimizer to perform back propagation or gradient descent (could also use SGD check torch.optim)
optimizer = optim.Adam(model.parameters(), lr=0.1)
#number of times bp/gd is performed on code(in this case 1000 runs)
epochs = 1000


# Training the neural network
def train_model(model, losses, optimizer, epochs, x_train, y_train):
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = losses(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())


        #print(f"{epoch}, Loss: {loss.item()}")
    print(f'First: {range(epochs)[0]} Loss : {train_losses[0]}')
    print(f'Last : {range(epochs)[-1]} Loss : {train_losses[-1]}')

    return train_losses


# Train the model
train_losses = train_model(model, losses, optimizer, epochs, x_train, y_train)

# Plot the training loss
plt.plot(train_losses)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs')
plt.grid()
plt.show()


# Plot original function and predicted derivatives
with torch.no_grad():
    model.eval()
    predicted_derivatives = model(x_train)
    plt.scatter(x_train, predicted_derivatives, label='Predicted Derivatives')
    plt.plot(x_train, func_derivative(x_train), label='True Derivatives', color='red')
    plt.plot(x_train, function(x_train), label='function', color='green')
    plt.xlabel('x')
    plt.ylabel('f\'(x)')
    plt.title('Predicted Derivatives vs True Derivatives')
    plt.legend()
    plt.grid()
    plt.show()