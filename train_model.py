"""Train a lightweight CPU LSTM and a Logistic Regression baseline."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from model_utils import prepare_sequences

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

def metrics(y_true, pred, prob):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_at_risk": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_risk": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_risk": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None
    }

def train(csv_path, epochs=15, seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    df = pd.read_csv(csv_path)
    X, y, feature_names, cols_by_week, target_col = prepare_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # Fit scaling on training data only.
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    def scale(a):
        return scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape).astype(np.float32)
    X_train, X_test = scale(X_train), scale(X_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    start = time.perf_counter()
    history = model.fit(
        X_train, y_train, validation_split=0.15, epochs=epochs, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )], verbose=0
    )
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    probs = model.predict(X_test, verbose=0).ravel()
    # Warm-up and repeated batch inference timing; report average per sample.
    for _ in range(3):
        model.predict(X_test[:min(32, len(X_test))], verbose=0)
    reps = 20
    start_inf = time.perf_counter()
    for _ in range(reps):
        model.predict(X_test, verbose=0)
    inference_ms_per_sample = (time.perf_counter() - start_inf) * 1000 / (reps * len(X_test))
    preds = (probs >= 0.5).astype(int)
    lstm_metrics = metrics(y_test, preds, probs)

    # Fair non-sequential baseline: same weekly features, flattened into one row.
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train.reshape(len(X_train), -1), y_train)
    lr_prob = lr.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    lr_metrics = metrics(y_test, lr_pred, lr_prob)

    MODEL_DIR.mkdir(exist_ok=True); RESULT_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "lightweight_lstm.keras"
    model.save(model_path)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(lr, MODEL_DIR / "logistic_baseline.joblib")
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": feature_names, "weeks": X.shape[1],
            "target_column": target_col, "threshold": 0.5,
            "seed": seed, "training_rows": len(X_train), "test_rows": len(X_test),
            "synthetic_data_warning": "Check the source dataset and disclose any simulated features."
        }, f, indent=2)
    results = pd.DataFrame([
        {"model": "Lightweight LSTM", **lstm_metrics,
         "training_time_seconds": training_seconds,
         "model_size_mb": model_path.stat().st_size / (1024 * 1024),
         "cpu_inference_ms_per_sample": inference_ms_per_sample},
        {"model": "Logistic Regression", **lr_metrics,
         "training_time_seconds": None, "model_size_mb": None,
         "cpu_inference_ms_per_sample": None}
    ])
    results.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(history.history).to_csv(RESULT_DIR / "training_history.csv", index=False)
    pd.DataFrame({"actual": y_test, "lstm_probability": probs, "lstm_prediction": preds}).to_csv(
        RESULT_DIR / "test_predictions.csv", index=False
    )
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    pd.DataFrame(cm, index=["Actual_Safe", "Actual_At_Risk"],
                 columns=["Predicted_Safe", "Predicted_At_Risk"]).to_csv(RESULT_DIR / "confusion_matrix.csv")
    # Model-agnostic permutation importance: shuffle each feature across test students,
    # retaining its weekly sequence positions, and measure ROC-AUC reduction.
    importance_rows = []
    if len(np.unique(y_test)) == 2:
        baseline_auc = roc_auc_score(y_test, probs)
        rng = np.random.default_rng(seed)
        for j, feature in enumerate(feature_names):
            permuted = X_test.copy()
            order = rng.permutation(len(permuted))
            permuted[:, :, j] = permuted[order, :, j]
            perm_prob = model.predict(permuted, verbose=0).ravel()
            perm_auc = roc_auc_score(y_test, perm_prob)
            importance_rows.append({"feature": feature, "baseline_roc_auc": baseline_auc,
                                    "permuted_roc_auc": perm_auc,
                                    "importance_auc_drop": baseline_auc - perm_auc,
                                    "method": "test-set permutation importance"})
    pd.DataFrame(importance_rows).sort_values("importance_auc_drop", ascending=False).to_csv(
        RESULT_DIR / "feature_importance.csv", index=False
    )
    print(results.to_string(index=False))
    print(f"\nSaved model to: {model_path}")
    print(f"Results saved to: {RESULT_DIR / 'model_comparison.csv'}")
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data" / "synthetic_demo_data.csv"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args.csv, args.epochs, args.seed)
