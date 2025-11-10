import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Data Preprocessing 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),         # Ensure images are grayscale
    transforms.ToTensor(),                               # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)),                # Normalize to [-1,1]
    transforms.Lambda(lambda x: x.view(-1))              # Flatten 28x28 to 784
])

train_path = 'C:/Users/Srijan Bhushan/Documents/Srijan Files/Python/VS Code/mnist_png/training'
test_path = 'C:/Users/Srijan Bhushan/Documents/Srijan Files/Python/VS Code/mnist_png/testing'

train_set = datasets.ImageFolder(root=train_path, transform=transform)
test_set = datasets.ImageFolder(root=test_path, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)


# 2. Visualize Data

images, labels = next(iter(train_loader))
plt.imshow(images[0].view(28,28).numpy(), cmap='gray')  
plt.title(f"Label: {labels[0].item()}")              
plt.show()

# 3. Define Neural Network
class FCN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = FCN_MNIST()


# 4. Training Setup

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 15
train_acc = []
test_acc = []


# 5. Training and Evaluation

for epoch in range(num_epochs):
    # Training phase
    model.train()
    correct, total = 0, 0
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        correct += (preds == target).sum().item()
        total += target.size(0)
    train_acc_epoch = correct / total
    train_acc.append(train_acc_epoch)


    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    test_acc_epoch = correct / total
    test_acc.append(test_acc_epoch)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc_epoch:.4f}, Test Acc: {test_acc_epoch:.4f}")


# 6. Plot Accuracy vs Epoch

plt.plot(range(1, num_epochs + 1), train_acc, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MNIST Accuracy vs Epoch')
plt.legend()
plt.show()
