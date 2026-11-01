from torchvision.datasets import CIFAR100
CIFAR100(root='./data', train=True, download=True)
CIFAR100(root='./data', train=False, download=True)
print("done")
