import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Step 1: Load and Preprocess Data
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Rotate to augment data
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize data
])

# load dataset
train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# 1000 examples for training
train_data_small, _ = random_split(train_data, [1000, len(train_data) - 1000])
train_loader = DataLoader(train_data_small, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# Define model using sequential nn
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        # Linear Layers with ReLU activation
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    # forward is just a model call
    def forward(self, x):
        return self.model(x)


# Set model
model = FCNN()
# Set loss function
criterion = nn.CrossEntropyLoss()
# Set learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Train model
def train_model():
    model.train()
    for epoch in range(50):  # Train for epochs
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Back prop
            loss.backward()
            optimizer.step()
            # Calculate total loss
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")


train_model()


# Use model on test set
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {correct / total:.4f}")


test_model()
