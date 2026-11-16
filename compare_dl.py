import mlflow
import pandas as pd

mlflow.set_experiment("lab3_cifar100")
runs = mlflow.search_runs(experiment_names=["lab3_cifar100"])
runs = runs[runs["metrics.best_val_acc"].notna()]
rows = []
for _, r in runs.iterrows():
    rows.append({
        "run_id": r["run_id"],
        "mode": r["params.mode"],
        "seed": r["params.seed"],
        "batch_size": r["params.batch_size"],
        "n_params": r["params.n_params"],
        "best_val_acc": r["metrics.best_val_acc"],
        "acc_lo": r["metrics.acc_lo"],
        "acc_hi": r["metrics.acc_hi"],
        "time_sec": r["metrics.time_sec"],
        "peak_memory_mb": r["metrics.peak_memory_mb"],
    })
df = pd.DataFrame(rows)
df.to_csv("dl_metrics.csv", index=False)
print(df.to_string())
