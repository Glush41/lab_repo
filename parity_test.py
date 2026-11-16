import torch
import numpy as np
from dl_model import CIFAR100CNN

torch.manual_seed(42)
model = CIFAR100CNN()
model.eval()
x = torch.randn(8, 3, 32, 32)
with torch.no_grad():
    out_cpu = model(x).numpy()

if torch.cuda.is_available():
    model_gpu = CIFAR100CNN().cuda()
    model_gpu.load_state_dict(model.state_dict())
    model_gpu.eval()
    with torch.no_grad():
        out_gpu = model_gpu(x.cuda()).cpu().numpy()
    diff = np.abs(out_cpu - out_gpu).max()
    print(f"CPU vs GPU max diff: {diff:.8f}")
    assert diff <= 1e-5, f"parity failed: diff={diff}"
    print("parity passed (CPU vs GPU)")
else:
    print("GPU not available, checking CPU determinism")
    with torch.no_grad():
        out_cpu2 = model(x).numpy()
    diff = np.abs(out_cpu - out_cpu2).max()
    print(f"CPU vs CPU max diff: {diff:.8f}")
    assert diff <= 1e-5, f"determinism failed: diff={diff}"
    print("parity passed (CPU vs CPU)")
