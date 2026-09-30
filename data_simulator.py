"""Generate a clearly synthetic dataset for demonstrating the application.

This is NOT real student data and must not be reported as empirical institutional evidence.
The target is generated from a latent synthetic risk score; the target column itself is never
included among the model input features.
"""
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_PER_WEEK = [
    "attendance_pct", "lms_logins", "resource_accesses",
    "submission_delay_days", "assessment_score"
]

def make_demo_data(n=1200, seed=42, weeks=6):
    rng = np.random.default_rng(seed)
    # Latent propensity is used to generate both behavior and the outcome.
    # It is not exported as a feature, preventing direct target leakage.
    latent = rng.normal(0, 1, n)
    rows = {}
    for w in range(1, weeks + 1):
        progress = (w - 1) / max(weeks - 1, 1)
        rows[f"week{w}_attendance_pct"] = np.clip(
            82 - 11 * latent + rng.normal(0, 12, n) - 2 * progress, 0, 100
        )
        rows[f"week{w}_lms_logins"] = np.clip(
            np.round(5.5 - 1.3 * latent + rng.normal(0, 2, n)), 0, 20
        )
        rows[f"week{w}_resource_accesses"] = np.clip(
            np.round(12 - 2.2 * latent + rng.normal(0, 4, n)), 0, 50
        )
        rows[f"week{w}_submission_delay_days"] = np.clip(
            np.round(1.5 + 1.1 * latent + rng.normal(0, 1.5, n)), 0, 14
        )
        rows[f"week{w}_assessment_score"] = np.clip(
            68 - 10 * latent + rng.normal(0, 14, n), 0, 100
        )
    # Outcome depends on the latent synthetic risk variable, not on any exported target feature.
    logit = -0.15 + 1.05 * latent
    probability = 1 / (1 + np.exp(-logit))
    rows["At_Risk"] = (rng.random(n) < probability).astype(int)
    rows["Student_ID"] = [f"S{i:04d}" for i in range(1, n + 1)]
    df = pd.DataFrame(rows)
    ordered = ["Student_ID"] + [
        f"week{w}_{f}" for w in range(1, weeks + 1) for f in FEATURES_PER_WEEK
    ] + ["At_Risk"]
    return df[ordered]

if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "synthetic_demo_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data().to_csv(out, index=False)
    print(f"Created synthetic demonstration data: {out}")
    print("WARNING: synthetic data only; do not describe these rows as real students.")
