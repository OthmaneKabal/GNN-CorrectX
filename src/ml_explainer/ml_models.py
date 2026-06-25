"""Train ML models to explain correct vs incorrect GNN predictions."""

from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from .prepare_ml_data import prepare_data
except ImportError:
    from prepare_ml_data import prepare_data


DEFAULT_MODELS = [
    "random_forest",
    "xgboost",
    "logistic_regression",
    "extra_trees",
    "svm_linear",
    "svm_rbf",
    "hist_gradient_boosting",
    "mlp",
]
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
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed_ml,
            n_jobs=1,
            bootstrap=True,
            max_samples=0.8,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
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

    if model_name == "svm":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed_ml,
        )

    if model_name == "svm_linear":
        return SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=seed_ml,
        )

    if model_name == "svm_rbf":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed_ml,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed_ml,
        )

    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=0.01,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=1000,
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
            "n_estimators": [150, 300, 500, 800],
            "max_depth": [3, 5, 8, 10, 12],
            "min_samples_split": [5, 10, 20, 40],
            "min_samples_leaf": [5, 10, 20, 30, 50],
            "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
            "bootstrap": [True],
            "max_samples": [0.5, 0.7, 0.85, 1.0],
            "class_weight": ["balanced"],
            "ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
        }

    if model_name == "extra_trees":
        return {
            "n_estimators": [150, 300, 500, 800],
            "max_depth": [3, 5, 8, 10, 12],
            "min_samples_split": [5, 10, 20, 40],
            "min_samples_leaf": [5, 10, 20, 30, 50],
            "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
            "bootstrap": [False, True],
            "class_weight": ["balanced"],
            "ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
        }

    if model_name == "logistic_regression":
        return {
            "C": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": ["balanced"],
        }

    if model_name == "svm":
        return {
            "kernel": ["rbf"],
            "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03, 0.1],
            "class_weight": ["balanced"],
            "probability": [True],
        }

    if model_name == "svm_linear":
        return {
            "C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "class_weight": ["balanced"],
        }

    if model_name == "svm_rbf":
        return {
            "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03, 0.1],
            "class_weight": ["balanced"],
        }

    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_iter": [100, 200, 300, 500],
            "max_leaf_nodes": [7, 15, 31],
            "max_depth": [2, 3, 5, None],
            "min_samples_leaf": [10, 20, 30, 50],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
            "early_stopping": [True],
            "validation_fraction": [0.1, 0.15, 0.2],
        }

    if model_name == "mlp":
        return {
            "hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "learning_rate_init": [0.0003, 0.001, 0.003],
            "early_stopping": [True],
            "validation_fraction": [0.1, 0.15, 0.2],
            "max_iter": [1000],
        }

    if model_name == "xgboost":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.08],
            "subsample": [0.6, 0.75, 0.9],
            "colsample_bytree": [0.6, 0.75, 0.9],
            "min_child_weight": [3, 5, 10, 20],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0, 5.0],
            "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        }

    raise ValueError(f"Unknown model: {model_name}")


def _fit_model(model_name, X_train, y_train, seed_ml, tune, n_iter, cv, scoring, n_jobs):
    model = _build_model(model_name, seed_ml=seed_ml, y_train=y_train)
    if not tune:
        model.fit(X_train, y_train)
        return model, None

    _log(
        f"Tuning {model_name} with RandomizedSearchCV "
        f"(n_iter={n_iter}, cv={cv}, scoring={scoring})"
    )
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=seed_ml,
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=_param_distributions(model_name),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_strategy,
        random_state=seed_ml,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    _log(f"{model_name} best CV score: {search.best_score_:.4f}")
    _log(f"{model_name} best params: {search.best_params_}")
    return search.best_estimator_, {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "cv_results": _cv_result_rows(search),
    }


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _json_dumps(payload):
    return json.dumps(payload, default=_json_safe, sort_keys=True)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_safe)


def _cv_result_rows(search):
    cv_results = search.cv_results_
    rows = []
    params_list = cv_results.get("params", [])

    split_columns = sorted(
        column
        for column in cv_results
        if column.startswith("split") and column.endswith("_test_score")
    )

    for candidate_index, params in enumerate(params_list):
        row = {
            "candidate_index": candidate_index,
            "params": _json_dumps(params),
            "mean_cv_score": cv_results["mean_test_score"][candidate_index],
            "std_cv_score": cv_results["std_test_score"][candidate_index],
            "rank_cv_score": cv_results["rank_test_score"][candidate_index],
            "mean_fit_time": cv_results["mean_fit_time"][candidate_index],
            "std_fit_time": cv_results["std_fit_time"][candidate_index],
            "mean_score_time": cv_results["mean_score_time"][candidate_index],
            "std_score_time": cv_results["std_score_time"][candidate_index],
        }
        for split_column in split_columns:
            row[split_column] = cv_results[split_column][candidate_index]
        rows.append(row)

    return rows


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


def _select_best_model_on_val(metrics_df, selection_metric):
    if metrics_df.empty:
        return None

    val_metrics = metrics_df[metrics_df["split"] == "val"].copy()
    if val_metrics.empty:
        return None
    if selection_metric not in val_metrics.columns:
        raise ValueError(f"Selection metric not found in metrics: {selection_metric}")

    val_metrics = val_metrics.sort_values(
        by=[selection_metric, "mcc", "roc_auc"],
        ascending=False,
    )
    return val_metrics.iloc[0].to_dict()


def _save_confusion_matrix_image(row, output_dir, dataset_name):
    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    model_name = row["model"]
    split_name = row["split"]
    image_path = (
        output_dir
        / "confusion_matrices"
        / f"seed_{row['seed_gnn']}__{model_name}__{split_name}_cm.png"
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
    cv=5,
    scoring="f1_macro",
    selection_metric="f1_macro",
    n_jobs=1,
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
    all_performance_rows = []
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
                n_jobs=n_jobs,
            )
            trained_models[model_name] = model
            if tuning_result is not None:
                best_params[model_name] = {
                    "best_score": tuning_result.get("best_score"),
                    "best_params": tuning_result.get("best_params", {}),
                }
                for performance_row in tuning_result.get("cv_results", []):
                    row = dict(performance_row)
                    row.update(
                        {
                            "model": model_name,
                            "seed_gnn": seed_gnn,
                            "seed_ml": seed_ml,
                            "tuned": tune,
                            "scoring": scoring,
                        }
                    )
                    all_performance_rows.append(row)

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
    all_performance_df = pd.DataFrame(all_performance_rows)
    best_model_summary = _select_best_model_on_val(metrics_df, selection_metric)

    if save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(excel_path).stem
        metrics_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_metrics.xlsx"
        confusion_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__confusion_matrices.xlsx"
        failures_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_failures.json"
        best_params_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_best_params.json"
        best_model_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__best_model_on_val.json"
        all_performance_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__all_performance_ml_explainer.xlsx"
        aggregate_results_path = output_dir / f"{dataset_name}__seed_gnn_{seed_gnn}__ml_results.json"

        _log(f"Saving metrics: {metrics_path}")
        metrics_df.to_excel(metrics_path, index=False)
        _log(f"Saving confusion matrices: {confusion_path}")
        confusion_df.to_excel(confusion_path, index=False)
        if not all_performance_df.empty:
            _log(f"Saving all hyperparameter performance: {all_performance_path}")
            all_performance_df.to_excel(all_performance_path, index=False)
        elif all_performance_path.exists():
            _log(f"Removing stale all performance file: {all_performance_path}")
            all_performance_path.unlink()
        _log("Saving confusion matrix images")
        for _, confusion_row in confusion_df.iterrows():
            try:
                _save_confusion_matrix_image(confusion_row, output_dir, dataset_name)
            except OSError as error:
                _log(f"[WARNING] Could not save confusion matrix image: {error}")

        if failures:
            _log(f"Saving failures: {failures_path}")
            _write_json(failures_path, failures)
        elif failures_path.exists():
            _log(f"Removing stale failures file: {failures_path}")
            failures_path.unlink()

        if best_params:
            _log(f"Saving best params: {best_params_path}")
            _write_json(best_params_path, best_params)
        elif best_params_path.exists():
            _log(f"Removing stale best params file: {best_params_path}")
            best_params_path.unlink()

        if best_model_summary:
            _log(f"Saving best model selected on validation: {best_model_path}")
            _write_json(best_model_path, best_model_summary)
        elif best_model_path.exists():
            _log(f"Removing stale best model file: {best_model_path}")
            best_model_path.unlink()

        aggregate_results = {
            "dataset": dataset_name,
            "seed_gnn": seed_gnn,
            "seed_ml": seed_ml,
            "tuned": tune,
            "scoring": scoring,
            "selection_metric": selection_metric,
            "n_jobs": n_jobs,
            "metrics": metrics_df.to_dict(orient="records"),
            "confusion_matrices": confusion_df.to_dict(orient="records"),
            "all_performance": all_performance_df.to_dict(orient="records"),
            "best_params": best_params,
            "best_model_summary": best_model_summary,
            "failures": failures,
        }
        _log(f"Saving aggregate results JSON: {aggregate_results_path}")
        _write_json(aggregate_results_path, aggregate_results)

    return {
        "models": trained_models,
        "metrics": metrics_df,
        "confusion_matrices": confusion_df,
        "all_performance": all_performance_df,
        "best_params": best_params,
        "best_model_summary": best_model_summary,
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
        help=(
            "Comma-separated models: random_forest,xgboost,logistic_regression,"
            "extra_trees,svm,svm_linear,svm_rbf,hist_gradient_boosting,mlp."
        ),
    )
    parser.add_argument("--output-path", default=None, help="Output directory for metrics.")
    parser.add_argument("--no-save", action="store_true", help="Do not save metrics files.")
    parser.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV tuning.")
    parser.add_argument("--n-iter", default=25, type=int, help="Random search iterations.")
    parser.add_argument("--cv", default=5, type=int, help="Stratified cross-validation folds for tuning.")
    parser.add_argument("--scoring", default="f1_macro", help="Scoring metric for tuning.")
    parser.add_argument("--n-jobs", default=1, type=int, help="Parallel jobs for RandomizedSearchCV.")
    parser.add_argument(
        "--selection-metric",
        default="f1_macro",
        help="Validation metric used to select the final best model.",
    )
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
        selection_metric=args.selection_metric,
        n_jobs=args.n_jobs,
    )

    print(result["metrics"].to_string(index=False))
    if result["best_model_summary"]:
        print("\nBest model on validation:")
        print(json.dumps(result["best_model_summary"], indent=2))
    if result["failures"]:
        print("\nFailures:")
        print(json.dumps(result["failures"], indent=2))


if __name__ == "__main__":
    main()
