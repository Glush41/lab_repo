import numpy as np
from torchvision.datasets import CIFAR100
from skimage.feature import hog

def load_subset(root, per_class=100):
    ds = CIFAR100(root=root, train=True, download=False)
    indices = []
    for cls in range(100):
        cls_idx = [i for i, (_, label) in enumerate(ds) if label == cls][:per_class]
        indices.extend(cls_idx)
    images = np.stack([np.array(ds[i][0]) for i in indices])
    labels = np.array([ds[i][1] for i in indices])
    return images, labels

def hog_features(images, chunk_size=1000):
    feats = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start:start+chunk_size]
        for img in chunk:
            gray = np.mean(img, axis=2).astype(np.uint8)
            f = hog(gray, orientations=9, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), feature_vector=True)
            feats.append(f)
    return np.array(feats)
