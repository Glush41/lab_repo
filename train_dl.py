import argparse, time, tracemalloc, hashlib, copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
import mlflow, yaml
from dl_config_schema import DLConfig
from dl_model import CIFAR100CNN, count_params
from dl_data import get_full_dataloaders, get_stream_dataloaders

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_sum += criterion(out, y).item() * len(y)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, loss_sum / total, np.array(all_preds), np.array(all_labels)

def bootstrap_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append((y_true[idx] == y_pred[idx]).mean())
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def train_one(cfg: DLConfig, run_name: str):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    if cfg.mode == "full":
        train_loader, test_loader = get_full_dataloaders(cfg, transform)
    else:
        train_loader, test_loader = get_stream_dataloaders(cfg, transform)

    model = CIFAR100CNN().to(device)
    n_params = count_params(model)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment("lab3_cifar100")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg.model_dump())
        mlflow.log_param("n_params", n_params)

        tracemalloc.start()
        t0 = time.time()
        best_acc, best_state, patience = -1.0, None, 0
        history = {"train_loss": [], "val_acc": []}
        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y)
            train_loss = epoch_loss / len(train_loader.dataset)
            val_acc, _, _, _ = evaluate(model, test_loader, device)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            mlflow.log_metrics({"train_loss": train_loss, "val_acc": val_acc}, step=epoch)
            print(f"epoch {epoch+1}: loss={train_loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc, best_state, patience = val_acc, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print(f"early stop at epoch {epoch+1}")
                    break

        elapsed = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024**2

        model.load_state_dict(best_state)
        _, _, all_preds, all_labels = evaluate(model, test_loader, device)
        acc_lo, acc_hi = bootstrap_ci(all_labels, all_preds)
        torch.save(model.state_dict(), "dl_model.pt")
        model_hash = file_hash("dl_model.pt")
        mlflow.log_metrics({"best_val_acc": best_acc, "acc_lo": acc_lo, "acc_hi": acc_hi,
                            "time_sec": elapsed, "peak_memory_mb": peak_mb})
        mlflow.log_dict({"history": history}, "history.json")
        mlflow.log_dict({"model_sha256": model_hash}, "model_hash.json")
        mlflow.log_artifact("dl_model.pt")

        run_id = mlflow.active_run().info.run_id
        print(f"run_id={run_id} mode={cfg.mode} best_acc={best_acc:.4f} "
              f"[{acc_lo:.4f}, {acc_hi:.4f}] time={elapsed:.1f}s "
              f"peak_mem={peak_mb:.1f}MB hash={model_hash[:16]}...")
        return run_id, best_acc, elapsed, peak_mb, model_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = DLConfig(**yaml.safe_load(f))
    train_one(cfg, args.run_name or f"{cfg.mode}_seed{cfg.seed}")
