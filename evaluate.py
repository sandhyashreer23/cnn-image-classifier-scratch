import torch
from data import get_dataloaders
from model import SimpleCNN
import matplotlib.pyplot as plt

def evaluate():
    device = torch.device("cpu")

    _, test_loader = get_dataloaders(batch_size=64, subset_size=5000)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    model.eval()  # switches off dropout — we want the full network for evaluation

    correct = 0
    total = 0

    with torch.no_grad():  # no need to track gradients, we're not training here
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

def visualize_first_layer_filters():
    model = SimpleCNN()
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))

    filters = model.conv1.weight.data.clone()  # shape: [32, 3, 3, 3]

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        f = filters[i]
        f = (f - f.min()) / (f.max() - f.min())  # normalize to 0-1 for display
        ax.imshow(f.permute(1, 2, 0))  # rearrange channels for imshow
        ax.axis("off")
    plt.suptitle("First Conv Layer Filters")
    plt.savefig("conv1_filters.png")
    print("Saved filter visualization to conv1_filters.png")

if __name__ == "__main__":
    evaluate()
    visualize_first_layer_filters()

    import matplotlib.pyplot as plt
from gabor_filters import generate_gabor_filters

def visualize_gabor_filters():
    filters = generate_gabor_filters()  # shape [32, 3, 3, 3]

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        f = filters[i]
        f = (f - f.min()) / (f.max() - f.min())
        ax.imshow(f.permute(1, 2, 0))
        ax.axis("off")
    plt.suptitle("Gabor Filter Bank")
    plt.savefig("gabor_filters.png")
    print("Saved to gabor_filters.png")

if __name__ == "__main__":
    visualize_gabor_filters()