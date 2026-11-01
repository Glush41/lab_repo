import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100

ds = CIFAR100(root='./data', train=True, download=False)

indices = []
for cls in range(100):
    cls_indices = [i for i, (_, label) in enumerate(ds) if label == cls][:100]
    indices.extend(cls_indices)

images = np.stack([np.array(ds[i][0]) for i in indices])
labels = np.array([ds[i][1] for i in indices])

print(f"subset size: {len(indices)}, classes: {len(np.unique(labels))}")

for ch, name in enumerate(['R', 'G', 'B']):
    ch_data = images[:, :, :, ch]
    print(f"{name}: mean={ch_data.mean():.1f}, std={ch_data.std():.1f}, "
          f"min={ch_data.min()}, max={ch_data.max()}")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
class_names = ds.classes
for ax, cls_id in zip(axes.flat, range(10)):
    idx = indices[cls_id * 100]
    ax.imshow(np.array(ds[idx][0]))
    ax.set_title(f"{class_names[cls_id]}\n(n=100)", fontsize=8)
    ax.axis('off')
plt.suptitle("CIFAR-100: examples (32x32, 100 classes)")
plt.tight_layout()
plt.savefig("problem_plot.png", dpi=150)
print("plot saved")
