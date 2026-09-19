import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model_gabor import GaborCNN

def train_gabor():
    device = torch.device("cpu")
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(batch_size=64, subset_size=5000)

    model = GaborCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Only pass trainable parameters to the optimizer (conv1 is frozen, so it's skipped automatically)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

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

    torch.save(model.state_dict(), "trained_model_gabor.pth")
    print("Model saved to trained_model_gabor.pth")

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Gabor Model Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    train_gabor()