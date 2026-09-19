import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Normalize using CIFAR-10's known channel means/stds
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

def get_dataloaders(batch_size=64, subset_size=None):
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Optional: use a smaller subset for faster CPU training
    if subset_size:
        train_set = torch.utils.data.Subset(train_set, range(subset_size))
        test_set = torch.utils.data.Subset(test_set, range(subset_size // 5))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(subset_size=1000)
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Labels:", labels[:10])