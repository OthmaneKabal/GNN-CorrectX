"""Train ML models to explain correct vs incorrect GNN predictions."""

from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV

try:
    from .prepare_ml_data import prepare_data
except ImportError:
    from prepare_ml_data import prepare_data


DEFAULT_MODELS = ["random_forest", "xgboost", "logistic_regression", "extra_trees"]
DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_explainer" / "ml_models"


def _log(message):
    print(f"[ml_models] {message}")


def _scale_pos_weight(y_train):
    counts = y_train.value_counts()
    positives = counts.get(1, 0)
    negatives = counts.get(0, 0)
    if positives == 0:
        return 1.0
    return negatives / positives


def _build_model(model_name, seed_ml, y_train=None):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed_ml,
            n_jobs=1,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed_ml,
            n_jobs=1,
        )

    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed_ml,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "xgboost is not installed. Install it with `pip install xgboost` "
                "or remove xgboost from the model list."
            ) from error

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed_ml,
            n_jobs=1,
            scale_pos_weight=_scale_pos_weight(y_train) if y_train is not None else 1.0,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _param_distributions(model_name):
    if model_name == "random_forest":
        return {
            "n_estimators": [150, 300, 500],
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [2, 5, 10, 20],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "bootstrap": [True],
            "class_weight": ["balanced"],
        }

    if model_name == "extra_trees":
        return {
            "n_estimators": [150, 300, 500],
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [2, 5, 10, 20],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "bootstrap": [False, True],
            "class_weight": ["balanced"],
        }

    if model_name == "logistic_regression":
        return {
            "C": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": ["balanced"],
        }

    if model_name == "xgboost":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5, 10],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }

    raise ValueError(f"Unknown model: {model_name}")


def _fit_model(model_name, X_train, y_train, seed_ml, tune, n_iter, cv, scoring):
    model = _build_model(model_name, seed_ml=seed_ml, y_train=y_train)
    if not tune:
        model.fit(X_train, y_train)
        return model, None

    _log(
        f"Tuning {model_name} with RandomizedSearchCV "
        f"(n_iter={n_iter}, cv={cv}, scoring={scoring})"
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=_param_distributions(model_name),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed_ml,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    _log(f"{model_name} best CV score: {search.best_score_:.4f}")
    _log(f"{model_name} best params: {search.best_params_}")
    return search.best_estimator_, {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
    }


def _positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def _evaluate_model(model, X, y, split_name):
    y_pred = model.predict(X)
    y_score = _positive_scores(model, X)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "recall_class_0": recall_score(y, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
    }

    if y_score is not None and y.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y, y_score)
    else:
        metrics["roc_auc"] = None

    return metrics


def _confusion_matrix_row(model, X, y, split_name):
    y_pred = model.predict(X)
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "split": split_name,
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def _save_confusion_matrix_image(row, output_dir, dataset_name):
    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    model_name = row["model"]
    split_name = row["split"]
    image_path = (
        output_dir
        / "confusion_matrices"
        / f"{dataset_name}__seed_gnn_{row['seed_gnn']}__{model_name}__{split_name}_confusion_matrix.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{model_name} - {split_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for true_idx in range(2):
        for pred_idx in range(2):
            value = matrix[true_idx][pred_idx]
            ax.text(
                pred_idx,
                true_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max(max(matrix[0]), max(matrix[1])) / 2 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    return image_path


def train_ml_models(
    excel_path,
    seed_gnn,
    seed_ml=42,
    model_names=None,
    output_path=None,
    save_results=True,
    tune=False,
    n_iter=25,
    cv=3,
    scoring="f1_macro",
):
    """Train configured ML models on one GNN seed and evaluate train/val/test."""
    model_names = model_names or DEFAULT_MODELS

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        excel_path=excel_path,
        seed_gnn=seed_gnn,
        seed_ml=seed_ml,
    )

    output_dir = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_DIR
    metrics_rows = []
    confusion_rows = []
    best_params = {}
    trained_models = {}
    failures = {}

    for model_name in model_names:
        _log(f"Training model: {model_name}")
        try:
            model, tuning_result = _fit_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                seed_ml=seed_ml,
                tune=tune,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
            )
            trained_models[model_name] = model
            if tuning_result is not None:
                best_params[model_name] = tuning_result

            for split_name, X_split, y_split in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                row = _evaluate_model(model, X_split, y_split, split_name)
                row.update(
                    {
                        "model": model_name,
                        "seed_gnn": seed_gnn,
                        "seed_ml": seed_ml,
                        "tuned": tune,
                    }
                )
                metrics_rows.append(row)

                confusion_row = _confusion_matrix_row(model, X_split, y_split, split_name)
                confusion_row.update(
                    {
                        "model": model_name,
                        "seed_gnn": seed_gnn,
                        "seed_ml": seed_ml,
                        "tuned": tune,
                    }
                )
                confusion_rows.append(confusion_row)
        except Exception as error:
            failures[model_name] = str(error)
            _log(f"[ERROR] {model_name}: {error}")

    metrics_df = pd.DataFrame(metrics_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(excel_path).stem
        metrics_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_metrics.xlsx"
        confusion_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__confusion_matrices.xlsx"
        failures_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_failures.json"
        best_params_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_best_params.json"

        _log(f"Saving metrics: {metrics_path}")
        metrics_df.to_excel(metrics_path, index=False)
        _log(f"Saving confusion matrices: {confusion_path}")
        confusion_df.to_excel(confusion_path, index=False)
        _log("Saving confusion matrix images")
        for _, confusion_row in confusion_df.iterrows():
            _save_confusion_matrix_image(confusion_row, output_dir, dataset_name)

        if failures:
            _log(f"Saving failures: {failures_path}")
            with open(failures_path, "w", encoding="utf-8") as file:
                json.dump(failures, file, indent=2)
        elif failures_path.exists():
            _log(f"Removing stale failures file: {failures_path}")
            failures_path.unlink()

        if best_params:
            _log(f"Saving best params: {best_params_path}")
            with open(best_params_path, "w", encoding="utf-8") as file:
                json.dump(best_params, file, indent=2)
        elif best_params_path.exists():
            _log(f"Removing stale best params file: {best_params_path}")
            best_params_path.unlink()

    return {
        "models": trained_models,
        "metrics": metrics_df,
        "confusion_matrices": confusion_df,
        "best_params": best_params,
        "failures": failures,
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
    }


def _parse_models(text):
    if text is None or text.strip() == "":
        return DEFAULT_MODELS
    return [model.strip() for model in text.split(",") if model.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Train ML explainer models on a cleaned correct-prediction dataset."
    )
    parser.add_argument("--excel-path", required=True, help="Cleaned ML Excel path.")
    parser.add_argument("--seed-gnn", required=True, type=int, help="GNN seed to use.")
    parser.add_argument("--seed-ml", default=42, type=int, help="ML split/model seed.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated models: random_forest,xgboost,logistic_regression,extra_trees.",
    )
    parser.add_argument("--output-path", default=None, help="Output directory for metrics.")
    parser.add_argument("--no-save", action="store_true", help="Do not save metrics files.")
    parser.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV tuning.")
    parser.add_argument("--n-iter", default=25, type=int, help="Random search iterations.")
    parser.add_argument("--cv", default=3, type=int, help="Cross-validation folds for tuning.")
    parser.add_argument("--scoring", default="f1_macro", help="Scoring metric for tuning.")
    args = parser.parse_args()

    result = train_ml_models(
        excel_path=args.excel_path,
        seed_gnn=args.seed_gnn,
        seed_ml=args.seed_ml,
        model_names=_parse_models(args.models),
        output_path=args.output_path,
        save_results=not args.no_save,
        tune=args.tune,
        n_iter=args.n_iter,
        cv=args.cv,
        scoring=args.scoring,
    )

    print(result["metrics"].to_string(index=False))
    if result["failures"]:
        print("\nFailures:")
        print(json.dumps(result["failures"], indent=2))


if __name__ == "__main__":
    main()
