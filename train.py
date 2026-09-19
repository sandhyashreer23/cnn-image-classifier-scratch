import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import SimpleCNN

def train():
    device = torch.device("cpu")  # your machine has no CUDA/ROCm support in PyTorch
    print("Using device:", device)

    # Start with a subset for CPU speed — bump this up later if you have time
    train_loader, test_loader = get_dataloaders(batch_size=64, subset_size=5000)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved to trained_model.pth")

    return model, test_loader

if __name__ == "__main__":
    train()