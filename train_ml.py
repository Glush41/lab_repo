import argparse, time, tracemalloc, hashlib
import numpy as np
import mlflow, mlflow.sklearn, joblib, yaml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from config_schema import PipelineConfig
from data_utils import load_subset, hog_features

def bootstrap_ci(y_true, y_pred, metric_fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main(cfg: PipelineConfig):
    mlflow.set_experiment("lab2_cifar100")
    with mlflow.start_run(run_name=f"{cfg.mode}_{cfg.model_type}_chunk{cfg.chunk_size}"):
        mlflow.log_params(cfg.model_dump())
        tracemalloc.start()
        t0 = time.time()

        images, labels = load_subset(cfg.data_root, cfg.subset_per_class)
        X = hog_features(images, chunk_size=cfg.chunk_size)
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y)

        # Наивный ориентир
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

        # Масштабирование (fit на train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Модель: SGD (partial_fit) или RF
        if cfg.model_type == "sgd":
            model = SGDClassifier(loss="log_loss", alpha=cfg.sgd_alpha,
                                  random_state=cfg.seed, max_iter=1000)
            if cfg.mode == "full":
                model.fit(X_train, y_train)
            else:
                classes = np.arange(100)
                for start in range(0, len(X_train), cfg.chunk_size):
                    X_chunk = X_train[start:start+cfg.chunk_size]
                    y_chunk = y_train[start:start+cfg.chunk_size]
                    model.partial_fit(X_chunk, y_chunk, classes=classes)
        else:
            model = RandomForestClassifier(n_estimators=cfg.rf_n_estimators,
                                           random_state=cfg.seed, n_jobs=1)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        acc_lo, acc_hi = bootstrap_ci(y_test, y_pred, accuracy_score)

        # Confusion-пары (метрика акцента для P2)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(100))
        np.fill_diagonal(cm, 0)
        top_pairs = []
        cm_copy = cm.copy()
        for _ in range(5):
            i, j = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
            top_pairs.append({"true": int(i), "pred": int(j), "count": int(cm_copy[i, j])})
            cm_copy[i, j] = 0

        elapsed = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / 1024**2

        joblib.dump(model, "ml_model.joblib")
        model_hash = file_hash("ml_model.joblib")

        mlflow.log_metrics({
            "accuracy": acc, "dummy_accuracy": dummy_acc,
            "f1_macro": f1, "precision_macro": prec, "recall_macro": rec,
            "acc_lo": acc_lo, "acc_hi": acc_hi,
            "time_sec": elapsed, "peak_memory_mb": peak_mb
        })
        mlflow.log_dict({"top_confusion_pairs": top_pairs}, "confusion_pairs.json")
        mlflow.log_dict({"model_sha256": model_hash}, "model_hash.json")
        mlflow.log_artifact("ml_model.joblib")

        run_id = mlflow.active_run().info.run_id
        print(f"run_id={run_id} mode={cfg.mode} model={cfg.model_type} "
              f"acc={acc:.4f} dummy={dummy_acc:.4f} f1={f1:.4f} "
              f"time={elapsed:.1f}s peak_mem={peak_mb:.1f}MB hash={model_hash[:16]}...")
        print("top_confusion_pairs:", top_pairs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = PipelineConfig(**yaml.safe_load(f))
    main(cfg)
