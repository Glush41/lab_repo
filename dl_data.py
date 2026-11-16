import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import CIFAR100

class StreamDataset(Dataset):
    def __init__(self, root, indices, train=True, transform=None):
        self.ds = CIFAR100(root=root, train=train, download=False)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.ds[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_subset_indices(root, per_class=100):
    ds = CIFAR100(root=root, train=True, download=False)
    indices = []
    for cls in range(100):
        cls_idx = [i for i, (_, label) in enumerate(ds) if label == cls][:per_class]
        indices.extend(cls_idx)
    return indices

def get_full_dataloaders(cfg, transform):
    ds = CIFAR100(root=cfg.data_root, train=True, download=False)
    indices = get_subset_indices(cfg.data_root, cfg.subset_per_class)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(indices)
    images = np.stack([np.array(ds[i][0]) for i in indices])
    labels = np.array([ds[i][1] for i in indices])
    X = torch.stack([transform(img) for img in images])
    y = torch.tensor(labels, dtype=torch.long)
    n = len(y)
    n_train = int(n * 0.8)
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    test_ds = TensorDataset(X[n_train:], y[n_train:])
    return (DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False))

def get_stream_dataloaders(cfg, transform):
    indices = get_subset_indices(cfg.data_root, cfg.subset_per_class)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * 0.8)
    train_ds = StreamDataset(cfg.data_root, indices[:n_train], train=True, transform=transform)
    test_ds = StreamDataset(cfg.data_root, indices[n_train:], train=True, transform=transform)
    return (DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers),
            DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers))
