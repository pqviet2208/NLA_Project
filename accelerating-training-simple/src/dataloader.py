from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_cifar(train_batch_size, eval_batch_size, num_workers=8, root="./data/cifar/"):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(root=root, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
