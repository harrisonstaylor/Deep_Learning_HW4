import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import time
import os
from tempfile import TemporaryDirectory


def evaluate_model_on_test(model, dataloader):
    # Compute acc on test set
    model.eval()
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    test_accuracy = running_corrects.double() / total_samples
    print(f'Test Accuracy: {test_accuracy:.4f}')


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        # Output epoch
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                # Train and evaluate
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()
                # Calc loss and accuracy and output
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Save best parameters from validation set
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model


# normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2860], [0.3530]),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.2860], [0.3530]),
])
# Download train and test sets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_train,
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
# Split train and val
train_ratio = 0.8
train_size = int(train_ratio * len(training_data))
val_size = len(training_data) - train_size

dataset_sizes = {'train': train_size, 'val': val_size}

train_set, val_set = random_split(training_data, [train_size, val_size])
# Init dataloaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

dataloaders = {'train': train_loader, 'val': val_loader}
# Train with cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and modify the model
model_conv = models.resnet18(weights='IMAGENET1K_V1')

# Freeze layers
for param in model_conv.parameters():
    param.requires_grad = False

# modify first layer to one channel
model_conv.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# average over the rgb channels for first layer
with torch.no_grad():
    model_conv.conv1.weight = nn.Parameter(model_conv.conv1.weight.sum(dim=1, keepdim=True))

# modify output to ten classes
model_conv.fc = nn.Linear(model_conv.fc.in_features, 10)

# specifically unfreeze layer 4
for name, param in model_conv.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

# High lr to fine tune
optimizer_conv = optim.SGD(filter(lambda p: p.requires_grad, model_conv.parameters()), lr=0.01, momentum=0.9)

# makes model update learning rate every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)

# trains and evaluates
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=10)
evaluate_model_on_test(model_conv, test_dataloader)

# saves model
torch.save(model_conv.state_dict(), 'fashion_mnist_resnet18.pth')
