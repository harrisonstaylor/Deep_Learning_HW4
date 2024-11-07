import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter


# load data
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # rotate data
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize data
])

# load data
train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 1000 examples for training
train_data_small, _ = random_split(train_data, [1000, len(train_data) - 1000])
train_loader = DataLoader(train_data_small, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# FCNN model
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

# init the model, criterion, and optimizer
criterion = nn.CrossEntropyLoss()

# train model and capture parameters
def train_and_params():
    parameter_trajectories = []
    loss_trajectories = []

    for run in range(2):  # Run SGD twice
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

        parameter_trajectories.append(parameters)
        loss_trajectories.append(losses)

    return parameter_trajectories, loss_trajectories

# parameter trajectories and loss
parameter_trajectories, loss_trajectories = train_and_params()

# Flatten stack all parameter vectors
all_params = np.vstack([np.array(traj) for traj in parameter_trajectories])
pca = PCA(n_components=2)
pca_params = pca.fit_transform(all_params)  # PCA on parameters
# Plot SGD trajectories and loss surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory
colors = ['b', 'r']  # Color for each
for idx, (params, losses) in enumerate(zip(parameter_trajectories, loss_trajectories)):
    # project parameters
    pca_proj = pca.transform(params)

    # Plot epochs as points
    ax.scatter(pca_proj[:, 0], pca_proj[:, 1], losses, color=colors[idx], label=f'Trajectory {idx + 1}', s=20)

# grid of points in 2D for surface
x = np.linspace(pca_proj[:, 0].min() - 20.5, pca_proj[:, 0].max() + 1.5, 25)
y = np.linspace(pca_proj[:, 1].min() - 3.5, pca_proj[:, 1].max() + 3.5, 25)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Calculate loss at each point
for i in range(25):
    for j in range(25):
        grid_point = np.array([X[i, j], Y[i, j]])

        # transform grid back to high dimensional space
        full_params = pca.inverse_transform(grid_point.reshape(1, -1))

        # reinit model
        model = FCNN()
        start_idx = 0
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()  # count of elements in parameter
                param_data = full_params[0, start_idx:start_idx + numel]

                # Convert the slice to a tensor and reshape
                param_tensor = torch.tensor(param_data, dtype=torch.float32).view_as(param)
                # Copy data in model's parameters
                param.copy_(param_tensor)

                # Increment starting index
                start_idx += numel

        # Calculate loss
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        Z[i, j] = total_loss / len(train_loader)

# Normalize loss
Z_min = Z.min()
Z_max = Z.max()
print(Z_min, Z_max)

# Plot
ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor = "none", color='gray', alpha=0.5, rstride=1, cstride=1)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Cross-Entropy Loss')
ax.legend()
plt.show()
