"""XGBoost context-window ablation used for the manuscript's feature-context table.

Runs LOSO folds for: single epoch, single epoch z-scored within recording,
context +/-2, +/-4, +/-8, and context +/-4 z-scored within recording.
Check the paper's reported numbers against this script's output before submission;
recording-wise standardisation uses signal features only, never labels.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from train_context import build_context_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def zscore_by_recording(X, recording):
    Z = np.empty_like(X, dtype=float)
    for rec in np.unique(recording):
        idx = np.flatnonzero(recording == rec)
        Z[idx] = StandardScaler().fit_transform(X[idx])
    return Z


def model():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, tree_method="hist",
                         n_jobs=-1, random_state=42, eval_metric="mlogloss")


def evaluate(name, X, y, groups):
    true, pred, subject_rows, folds = [], [], [], []
    started = time.time()
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups), 1):
        clf = model()
        clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        p = clf.predict(X[te]).astype(int)
        true.extend(y[te].tolist()); pred.extend(p.tolist())
        subject_rows.extend(groups[te].tolist())
        folds.append({"fold": fold, "held_out_subject": str(groups[te][0]),
                      "accuracy": float(accuracy_score(y[te], p)),
                      "macro_f1": float(f1_score(y[te], p, average="macro", zero_division=0)),
                      "cohen_kappa": float(cohen_kappa_score(y[te], p))})
    row = {"configuration": name, "features": int(X.shape[1]),
           "accuracy": float(accuracy_score(true, pred)),
           "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
           "cohen_kappa": float(cohen_kappa_score(true, pred)),
           "seconds": round(time.time()-started, 1)}
    pd.DataFrame({"subject": subject_rows, "y_true": true, "y_pred": pred}).to_csv(
        OUT / f"predictions_ablation_{name.lower().replace(' ', '_').replace('+','p').replace('±','pm')}.csv", index=False)
    return row, folds


def main():
    d = np.load(OUT / "features.npz", allow_pickle=True)
    X, y, groups = d["X"], d["y"].astype(int), d["groups"].astype(str)
    meta = pd.read_csv(OUT / "epoch_metadata.csv")
    if not np.array_equal(meta["label"].to_numpy(), y):
        raise ValueError("Metadata labels do not align with features.npz")
    rec = meta["recording"].astype(str).to_numpy()
    configs = [("Single epoch", X.copy()),
               ("Single z-scored", zscore_by_recording(X, rec))]
    for radius in (2, 4, 8):
        shifts = tuple(i for i in range(-radius, radius+1) if i != 0)
        configs.append((f"Context ±{radius}", build_context_features(X, rec, shifts=shifts)))
    c4 = build_context_features(X, rec, shifts=tuple(range(-4, 0)) + tuple(range(1, 5)))
    configs.append(("Context ±4 z-scored", zscore_by_recording(c4, rec)))
    rows, fold_results = [], {}
    for name, features in configs:
        row, folds = evaluate(name, features, y, groups)
        rows.append(row); fold_results[name] = folds
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "xgboost_context_ablation.csv", index=False)
    (OUT / "xgboost_context_ablation_folds.json").write_text(json.dumps(fold_results, indent=2))

if __name__ == "__main__":
    main()
