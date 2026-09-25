"""Machine-Learned Pairwise Matcher and Ranker (EXP-003).

Trains LightGBM GBDT on 1.35M leak-free training pairs with early stopping on the
S1-isolated validation split.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = os.path.join(REPO_ROOT, "work")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "name_tok_jac",
    "name_contain_q",
    "name_contain_c",
    "name_char2",
    "name_char3",
    "addr_tok_jac",
    "addr_contain_q",
    "addr_contain_c",
    "addr_char2",
    "addr_char3",
    "num_overlap",
    "rare_shared",
    "pool_name_eq",
    "pool_addr_eq",
    "pool_both_eq",
    "pool_ntok_eq",
    "pool_atok_eq",
    "pool_addr_any_missing",
    "is_us",
]


def prepare_datasets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pair pool and split cleanly into train and validation matrices by S1 ID."""
    pool_path = os.path.join(WORK_DIR, "pair_pool.parquet")
    val_split_path = os.path.join(WORK_DIR, "splits", "val_s1_ids.parquet")

    print("Loading pair pool and validation IDs...", flush=True)
    pool_df = pl.read_parquet(pool_path)
    val_df = pl.read_parquet(val_split_path)
    val_s1_set = set(val_df["s1_id"].to_list())

    # Add engineered features
    df = pool_df.with_columns([
        (pl.col("country") == "US").cast(pl.Float32).alias("is_us"),
        pl.col("pool_name_eq").cast(pl.Float32),
        pl.col("pool_addr_eq").cast(pl.Float32),
        pl.col("pool_both_eq").cast(pl.Float32),
        pl.col("pool_ntok_eq").cast(pl.Float32),
        pl.col("pool_atok_eq").cast(pl.Float32),
        pl.col("pool_addr_any_missing").cast(pl.Float32),
        pl.col("num_overlap").cast(pl.Float32),
        pl.col("rare_shared").cast(pl.Float32),
    ])

    # Partition by S1 ID (strict cluster-level isolation)
    train_df = df.filter(~pl.col("s1_id").is_in(list(val_s1_set)))
    val_df = df.filter(pl.col("s1_id").is_in(list(val_s1_set)))

    print(f"Train split: {train_df.height:,} pairs ({train_df['label'].sum():,} pos, {(train_df['label']==0).sum():,} neg)", flush=True)
    print(f"Val split: {val_df.height:,} pairs ({val_df['label'].sum():,} pos, {(val_df['label']==0).sum():,} neg)", flush=True)

    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df["label"].to_numpy().astype(np.int32)

    X_val = val_df.select(FEATURE_COLS).to_numpy()
    y_val = val_df["label"].to_numpy().astype(np.int32)

    return X_train, y_train, X_val, y_val


def train_matcher(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
) -> lgb.LGBMClassifier:
    """Train LightGBM Pairwise Classifier."""
    X_train, y_train, X_val, y_val = prepare_datasets()

    print(f"Training LightGBM Classifier (leaves={num_leaves}, lr={learning_rate}, n_est={n_estimators})...", flush=True)
    t0 = time.time()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)],
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Model trained in {time.time()-t0:.1f}s. Validation Pairwise ROC-AUC: {val_auc:.5f}", flush=True)

    # Print feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importances:", flush=True)
    for idx in sorted_idx:
        print(f"  {FEATURE_COLS[idx]:<25}: {importances[idx]}", flush=True)

    # Save model
    model_path = os.path.join(MODELS_DIR, "lgbm_matcher.joblib")
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}", flush=True)

    return model


if __name__ == "__main__":
    train_matcher()
