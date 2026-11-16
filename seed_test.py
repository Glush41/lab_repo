import torch
from dl_model import CIFAR100CNN

torch.manual_seed(42)
m1 = CIFAR100CNN()
x = torch.randn(1, 3, 32, 32)
out1 = m1(x).detach().numpy()

m2 = CIFAR100CNN()
out2 = m2(x).detach().numpy()

diff = abs(out1 - out2).max()
print(f"max diff without seed: {diff:.6f}")
assert diff > 1e-5, "seed test failed: outputs identical without seed"
print("seed test passed: без фиксации зерна результаты расходятся")
