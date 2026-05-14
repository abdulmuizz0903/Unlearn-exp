import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_dataloaders(batch_size=64, forget_digit=3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Base loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create subsets for forget and retain
    train_targets = train_dataset.targets
    forget_indices = (train_targets == forget_digit).nonzero(as_tuple=True)[0]
    retain_indices = (train_targets != forget_digit).nonzero(as_tuple=True)[0]

    forget_dataset = Subset(train_dataset, forget_indices)
    retain_dataset = Subset(train_dataset, retain_indices)

    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    
    # Same for test set to evaluate properly
    test_targets = test_dataset.targets
    test_forget_indices = (test_targets == forget_digit).nonzero(as_tuple=True)[0]
    test_retain_indices = (test_targets != forget_digit).nonzero(as_tuple=True)[0]
    
    test_forget_dataset = Subset(test_dataset, test_forget_indices)
    test_retain_dataset = Subset(test_dataset, test_retain_indices)
    
    test_forget_loader = DataLoader(test_forget_dataset, batch_size=batch_size, shuffle=False)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, forget_loader, retain_loader, test_forget_loader, test_retain_loader
