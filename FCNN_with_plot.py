import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load and Preprocess Data
print("Loading and preprocessing data...")
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Rotate to augment data
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize data
])

# Load dataset
train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 1000 examples for training
train_data_small, _ = random_split(train_data, [1000, len(train_data) - 1000])
train_loader = DataLoader(train_data_small, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print("Data loaded and preprocessed.")

# Define the fully connected neural network model
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model, criterion, and optimizer
criterion = nn.CrossEntropyLoss()

# Function to train the model and capture parameters
def train_and_capture_params():
    parameter_trajectories = []
    loss_trajectories = []

    for run in range(2):  # Run SGD twice
        print(f"Starting training run {run + 1}...")
        model = FCNN()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        parameters = []
        losses = []

        model.train()
        for epoch in range(50):
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Capture parameters and loss at the end of each epoch
            parameters.append(torch.cat([param.flatten() for param in model.parameters()]).detach().numpy())
            losses.append(total_loss / len(train_loader))
            print(f"Run {run + 1}, Epoch {epoch + 1}: Loss = {losses[-1]}")

        parameter_trajectories.append(parameters)
        loss_trajectories.append(losses)
        print(f"Completed training run {run + 1}.")

    return parameter_trajectories, loss_trajectories

# Get parameter trajectories and losses
print("Starting parameter capture...")
parameter_trajectories, loss_trajectories = train_and_capture_params()
print("Parameter capture complete.")

# Flatten and concatenate all parameter vectors for PCA
print("Performing PCA on parameter trajectories...")
all_params = np.vstack([np.array(traj) for traj in parameter_trajectories])
pca = PCA(n_components=2)
pca_params = pca.fit_transform(all_params)  # Perform PCA on concatenated parameters
print("PCA complete.")

# Plot the SGD trajectories and loss surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory as points
colors = ['b', 'r']  # Color for each initialization
for idx, (params, losses) in enumerate(zip(parameter_trajectories, loss_trajectories)):
    # Project each epoch's parameters into 2D PCA space
    pca_proj = pca.transform(params)

    # Plot each epoch's position in PCA space as points rather than a continuous line
    ax.scatter(pca_proj[:, 0], pca_proj[:, 1], losses, color=colors[idx], label=f'Trajectory {idx + 1}', s=20)
    print(f"Plotted trajectory {idx + 1} as points.")

# Dense grid of points in 2D PCA space for surface plot
print("Creating refined loss surface grid...")
x = np.linspace(pca_proj[:, 0].min() - 0.5, pca_proj[:, 0].max() + 0.5, 50)
y = np.linspace(pca_proj[:, 1].min() - 0.5, pca_proj[:, 1].max() + 0.5, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute loss at each point on the grid
print("Calculating refined loss surface...")
for i in range(50):
    for j in range(50):
        # Inverse transform to get back to full parameter space
        grid_point = np.array([X[i, j], Y[i, j]])
        full_params = torch.tensor(pca.inverse_transform(grid_point)).float()

        # Load parameters into model
        start_idx = 0
        model = FCNN()  # Reinitialize to ensure consistent state for each point
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()
                param.copy_(full_params[start_idx:start_idx + numel].view_as(param))
                start_idx += numel

        # Calculate cross-entropy loss for this grid point
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        Z[i, j] = total_loss / len(train_loader)
print("Refined loss surface calculation complete.")

# Plot the loss surface
ax.plot_surface(X, Y, Z, color='gray', alpha=0.5, rstride=1, cstride=1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Cross-Entropy Loss')
ax.legend()
plt.show()
print("Plotting complete.")
