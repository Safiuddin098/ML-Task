import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match RESNET-50 input size
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load the STL-10 dataset
train_dataset = STL10(root='./data', split='train', transform=transform, download=True)
test_dataset = STL10(root='./data', split='test', transform=transform, download=True)

# Define data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the RESNET-50 model
resnet50 = torchvision.models.resnet50(pretrained=True)
num_classes = 10  # Number of classes in STL-10
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=0.001)

# Training loop
num_epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

for epoch in range(num_epochs):
    resnet50.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation
resnet50.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = resnet50(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy}%")
