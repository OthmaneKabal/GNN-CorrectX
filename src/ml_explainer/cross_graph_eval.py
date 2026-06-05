"""Exploratory cross-graph ML evaluation."""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .ml_models import _build_model, _evaluate_model, _confusion_matrix_row
except ImportError:
    from ml_models import _build_model, _evaluate_model, _confusion_matrix_row


DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_explainer" / "cross_graph_eval"


def _log(message):
    print(f"[cross_graph_eval] {message}")


def _load_seed_data(excel_path, seed_gnn):
    df = pd.read_excel(excel_path)
    required_columns = {"seed", "correct_prediction"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {excel_path}: {sorted(missing_columns)}")

    df = df[df["seed"] == seed_gnn].copy()
    if df.empty:
        raise ValueError(f"No rows found for seed_gnn={seed_gnn} in {excel_path}")

    y = df["correct_prediction"].astype(int)
    X = df.drop(columns=["correct_prediction", "seed"])
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Feature matrix contains missing values in {excel_path}: {missing_counts}")

    return X, y


def _align_features(X_train, X_target):
    common_features = [column for column in X_train.columns if column in X_target.columns]
    missing_in_target = sorted(set(X_train.columns) - set(X_target.columns))
    extra_in_target = sorted(set(X_target.columns) - set(X_train.columns))

    if not common_features:
        raise ValueError("No common feature columns between train and target datasets.")

    return (
        X_train[common_features].copy(),
        X_target[common_features].copy(),
        common_features,
        missing_in_target,
        extra_in_target,
    )


def _short_dataset_name(path):
    stem = Path(path).stem
    return stem.replace("__correct_prediction_all_seeds_cleaned_ml", "")


def cross_graph_evaluate(
    train_excel_path,
    target_excel_path,
    train_seed_gnn,
    target_seed_gnn,
    model_name="random_forest",
    seed_ml=42,
    output_path=None,
):
    """Train on one graph/seed and evaluate on another graph/seed."""
    _log(f"Train dataset: {train_excel_path}")
    X_train, y_train = _load_seed_data(train_excel_path, train_seed_gnn)
    _log(f"Train rows: {len(X_train)} | class distribution: {y_train.value_counts().to_dict()}")

    _log(f"Target dataset: {target_excel_path}")
    X_target, y_target = _load_seed_data(target_excel_path, target_seed_gnn)
    _log(f"Target rows: {len(X_target)} | class distribution: {y_target.value_counts().to_dict()}")

    X_train, X_target, common_features, missing_in_target, extra_in_target = _align_features(
        X_train,
        X_target,
    )
    _log(f"Common features: {len(common_features)}")
    if missing_in_target:
        _log(f"Features ignored because missing in target: {missing_in_target}")
    if extra_in_target:
        _log(f"Target-only features ignored: {extra_in_target}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=common_features,
        index=X_train.index,
    )
    X_target_scaled = pd.DataFrame(
        scaler.transform(X_target),
        columns=common_features,
        index=X_target.index,
    )

    model = _build_model(model_name, seed_ml=seed_ml, y_train=y_train)
    _log(f"Training model: {model_name}")
    model.fit(X_train_scaled, y_train)

    train_metrics = _evaluate_model(model, X_train_scaled, y_train, "train_graph")
    target_metrics = _evaluate_model(model, X_target_scaled, y_target, "target_graph")
    metrics_df = pd.DataFrame(
        [
            {**train_metrics, "model": model_name, "seed_ml": seed_ml},
            {**target_metrics, "model": model_name, "seed_ml": seed_ml},
        ]
    )

    confusion_df = pd.DataFrame(
        [
            {
                **_confusion_matrix_row(model, X_train_scaled, y_train, "train_graph"),
                "model": model_name,
                "seed_ml": seed_ml,
            },
            {
                **_confusion_matrix_row(model, X_target_scaled, y_target, "target_graph"),
                "model": model_name,
                "seed_ml": seed_ml,
            },
        ]
    )

    output_dir = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    train_name = _short_dataset_name(train_excel_path)
    target_name = _short_dataset_name(target_excel_path)
    output_file = output_dir / f"{train_name}__to__{target_name}__{model_name}_metrics.xlsx"

    _log(f"Saving output: {output_file}")
    with pd.ExcelWriter(output_file) as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        confusion_df.to_excel(writer, sheet_name="confusion_matrix", index=False)
        pd.DataFrame({"feature": common_features}).to_excel(
            writer,
            sheet_name="features",
            index=False,
        )

    _log("Done")
    return {
        "metrics": metrics_df,
        "confusion_matrix": confusion_df,
        "features": common_features,
        "output_file": output_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Train on one graph and evaluate on another.")
    parser.add_argument("--train-excel-path", required=True)
    parser.add_argument("--target-excel-path", required=True)
    parser.add_argument("--train-seed-gnn", required=True, type=int)
    parser.add_argument("--target-seed-gnn", required=True, type=int)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--seed-ml", default=42, type=int)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    result = cross_graph_evaluate(
        train_excel_path=args.train_excel_path,
        target_excel_path=args.target_excel_path,
        train_seed_gnn=args.train_seed_gnn,
        target_seed_gnn=args.target_seed_gnn,
        model_name=args.model,
        seed_ml=args.seed_ml,
        output_path=args.output_path,
    )
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
