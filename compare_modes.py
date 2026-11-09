import mlflow
import pandas as pd

mlflow.set_experiment("lab2_cifar100")
runs = mlflow.search_runs(experiment_names=["lab2_cifar100"])
rows = []
for _, r in runs.iterrows():
    rows.append({
        "run_id": r["run_id"],
        "mode": r["params.mode"],
        "model_type": r["params.model_type"],
        "chunk_size": r["params.chunk_size"],
        "accuracy": r["metrics.accuracy"],
        "dummy_accuracy": r["metrics.dummy_accuracy"],
        "acc_lo": r["metrics.acc_lo"],
        "acc_hi": r["metrics.acc_hi"],
        "f1_macro": r["metrics.f1_macro"],
        "time_sec": r["metrics.time_sec"],
        "peak_memory_mb": r["metrics.peak_memory_mb"],
    })
df = pd.DataFrame(rows)
df.to_csv("ml_metrics.csv", index=False)
print(df.to_string())
