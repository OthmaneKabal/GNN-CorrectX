"""Recompute selected XGBoost/LightGBM OOF metrics with explicit F1 variants.

This reruns the best global hyperparameter configuration for xgboost and
lightgbm only, excludes KG-GEN, and reports accuracy, F1-macro, F1-weighted,
F1-class-0, and F1-class-1 from explicit out-of-fold predictions.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREPROCESSED_DIR = ROOT / "outputs" / "ml_explainer" / "preprocessed"
DOMAIN_SEED_PATH = ROOT / "outputs" / "reproducibility" / "representative_gnn_seeds.xlsx"
SUMMARY_PATH = (
    ROOT / "outputs" / "reproducibility" / "meta_model_hyperparameter_search"
    / "global_hyperparam_oof_summary.xlsx"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "reproducibility" / "selected_gradient_boosting_metrics.xlsx"
)

DEFAULT_ML_SEEDS = [42, 123, 456, 789, 2024]
MODEL_NAMES = ["xgboost", "lightgbm"]
DROP_FEATURES = [
    "total_degree",
    "nb_diverse_relations_total",
    "nb_is_a_total",
]
EXCLUDED_GRAPHS = {"biomed_tdg_kggen"}
GRAPH_LABELS = {
    "biomed_cg": "Biomed-CG",
    "biomed_tdg_gt2kg": "Biomed-TDG",
    "dbpedia_174_cg": "DBpedia-CG",
    "dbpedia_174_tdg": "DBpedia-TDG",
}
GRAPH_ORDER = ["Biomed-CG", "Biomed-TDG", "DBpedia-CG", "DBpedia-TDG"]
ARCH_ORDER = ["RGCN", "RotateEGCN"]


def _log(message):
    print(f"[recompute_boosters] {message}", flush=True)


def _parse_int_list(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_list(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _domain_from_graph(graph_name):
    if str(graph_name).startswith("biomed"):
        return "biomedical"
    if str(graph_name).startswith("dbpedia"):
        return "dbpedia"
    raise ValueError(f"Cannot infer domain from graph name: {graph_name}")


def _load_domain_seeds(path):
    df = pd.read_excel(path)
    return {row["domain"]: int(row["seed"]) for _, row in df.iterrows()}


def _load_best_configs(summary_path, model_names):
    df = pd.read_excel(summary_path)
    df = df[df["ml_model"].isin(model_names)].copy()
    if df.empty:
        raise ValueError(f"No selected model rows found for {model_names}: {summary_path}")
    best = df.sort_values(["ml_model", "rank"]).groupby("ml_model", as_index=False).first()
    configs = {}
    for _, row in best.iterrows():
        configs[row["ml_model"]] = {
            "config_id": row["config_id"],
            "hyperparams": json.loads(row["hyperparams_json"]),
            "hyperparams_json": row["hyperparams_json"],
            "rank": int(row["rank"]),
            "selection_f1_macro_mean": row["f1_macro_mean"],
            "selection_f1_macro_std": row["f1_macro_std"],
        }
    missing = sorted(set(model_names) - set(configs))
    if missing:
        raise ValueError(f"Missing best configs for: {missing}")
    return configs


def _discover_datasets(preprocessed_dir, domain_seeds):
    rows = []
    pattern = "*__correct_prediction_all_seeds_cleaned_ml.xlsx"
    for path in sorted(Path(preprocessed_dir).glob(pattern)):
        name = path.name.replace("__correct_prediction_all_seeds_cleaned_ml.xlsx", "")
        graph_name, gnn_model = name.rsplit("__", 1)
        if graph_name in EXCLUDED_GRAPHS:
            continue
        domain = _domain_from_graph(graph_name)
        rows.append(
            {
                "domain": domain,
                "graph_name": graph_name,
                "graph_label": GRAPH_LABELS.get(graph_name, graph_name),
                "gnn_model": gnn_model,
                "dataset_name": f"{graph_name}__{gnn_model}",
                "seed_gnn": domain_seeds[domain],
                "path": path,
            }
        )
    return rows


def _load_dataset(path, seed_gnn):
    df = pd.read_excel(path)
    df = df[df["seed"] == seed_gnn].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for seed_gnn={seed_gnn}: {path}")

    all_features = [column for column in df.columns if column not in {"correct_prediction", "seed"}]
    missing_drop = [feature for feature in DROP_FEATURES if feature not in all_features]
    if missing_drop:
        raise ValueError(f"{path} missing correlated features to drop: {missing_drop}")

    selected_features = [feature for feature in all_features if feature not in DROP_FEATURES]
    X = df[selected_features].apply(pd.to_numeric, errors="coerce")
    y = df["correct_prediction"].astype(int)
    if X.isna().any().any():
        missing_counts = X.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Feature matrix contains missing values: {missing_counts}")
    return X, y, selected_features


def _scale_pos_weight(y):
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def _build_model(model_name, params, seed, y_train, model_jobs):
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("Install xgboost first: pip install xgboost") from error
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=model_jobs,
            scale_pos_weight=_scale_pos_weight(y_train),
        )
        model.set_params(**params)
        return model

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("Install lightgbm first: pip install lightgbm") from error
        model = LGBMClassifier(
            random_state=seed,
            n_jobs=model_jobs,
            verbose=-1,
            class_weight="balanced",
        )
        model.set_params(**params)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def _positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return np.full(len(X), np.nan)


def _evaluate(y_true, y_pred, y_score):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_class_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_class_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if len(set(y_true)) > 1 and not np.isnan(y_score).all():
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def _run_oof(X, y, model_name, params, seed_ml, n_folds, model_jobs):
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_ml)
    oof_pred = pd.Series(index=y.index, dtype=int)
    oof_score = pd.Series(index=y.index, dtype=float)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = _build_model(
            model_name=model_name,
            params=params,
            seed=seed_ml + fold_idx,
            y_train=y_train,
            model_jobs=model_jobs,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_score = _positive_scores(model, X_test_scaled)
        oof_pred.iloc[test_idx] = y_pred
        oof_score.iloc[test_idx] = y_score

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **_evaluate(y_test, y_pred, y_score),
            }
        )

    predictions = pd.DataFrame(
        {
            "row_index": y.index,
            "y_true": y.astype(int),
            "y_pred": oof_pred.astype(int),
            "score_positive": oof_score.astype(float),
        }
    )
    return _evaluate(y, oof_pred.astype(int), oof_score.astype(float)), pd.DataFrame(fold_rows), predictions


def _mean_std_text(row, metric):
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def recompute(args):
    model_names = _parse_list(args.models)
    configs = _load_best_configs(args.summary_path, model_names)
    domain_seeds = _load_domain_seeds(args.domain_seed_path)
    datasets = _discover_datasets(args.preprocessed_dir, domain_seeds)
    ml_seeds = _parse_int_list(args.ml_seeds)

    details_rows = []
    fold_rows = []
    prediction_rows = []

    for model_name in model_names:
        config = configs[model_name]
        params = config["hyperparams"]
        _log(f"Model: {model_name}, config={config['config_id']}")
        for dataset in datasets:
            _log(
                f"  {dataset['graph_name']} / {dataset['gnn_model']} "
                f"(seed_gnn={dataset['seed_gnn']})"
            )
            X, y, selected_features = _load_dataset(dataset["path"], dataset["seed_gnn"])
            for seed_ml in ml_seeds:
                _log(f"    OOF seed_ml={seed_ml}")
                metrics, folds, predictions = _run_oof(
                    X=X,
                    y=y,
                    model_name=model_name,
                    params=params,
                    seed_ml=seed_ml,
                    n_folds=args.n_folds,
                    model_jobs=args.model_jobs,
                )
                details_rows.append(
                    {
                        **{key: value for key, value in dataset.items() if key != "path"},
                        "ml_model": model_name,
                        "config_id": config["config_id"],
                        "seed_ml": seed_ml,
                        "n_folds": args.n_folds,
                        "n_rows": len(y),
                        "n_features": len(selected_features),
                        "features": ",".join(selected_features),
                        **metrics,
                    }
                )
                for _, fold_row in folds.iterrows():
                    fold_rows.append(
                        {
                            **{key: value for key, value in dataset.items() if key != "path"},
                            "ml_model": model_name,
                            "config_id": config["config_id"],
                            "seed_ml": seed_ml,
                            **fold_row.to_dict(),
                        }
                    )
                predictions = predictions.assign(
                    domain=dataset["domain"],
                    graph_name=dataset["graph_name"],
                    graph_label=dataset["graph_label"],
                    gnn_model=dataset["gnn_model"],
                    dataset_name=dataset["dataset_name"],
                    seed_gnn=dataset["seed_gnn"],
                    ml_model=model_name,
                    config_id=config["config_id"],
                    seed_ml=seed_ml,
                )
                prediction_rows.append(predictions)

    details = pd.DataFrame(details_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    metric_columns = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_class_0",
        "f1_class_1",
        "recall_class_0",
        "recall_class_1",
        "mcc",
        "roc_auc",
    ]
    agg_spec = {
        "n_ml_seeds": ("seed_ml", "nunique"),
        "ml_seeds": ("seed_ml", lambda s: ",".join(map(str, sorted(s.unique())))),
        "n_rows": ("n_rows", "first"),
        "n_features": ("n_features", "first"),
    }
    for metric in metric_columns:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")

    aggregated = (
        details.groupby(
            ["domain", "graph_name", "graph_label", "gnn_model", "ml_model", "config_id", "seed_gnn"],
            as_index=False,
        )
        .agg(**agg_spec)
    )
    for metric in metric_columns:
        aggregated[f"{metric}_mean_std"] = aggregated.apply(
            lambda row, metric=metric: _mean_std_text(row, metric),
            axis=1,
        )

    aggregated["graph_label"] = pd.Categorical(
        aggregated["graph_label"], categories=GRAPH_ORDER, ordered=True
    )
    aggregated["gnn_model"] = pd.Categorical(
        aggregated["gnn_model"], categories=ARCH_ORDER, ordered=True
    )
    aggregated["ml_model"] = pd.Categorical(
        aggregated["ml_model"], categories=model_names, ordered=True
    )
    aggregated = aggregated.sort_values(["ml_model", "graph_label", "gnn_model"]).reset_index(drop=True)

    paper_table = aggregated[
        [
            "graph_label",
            "gnn_model",
            "ml_model",
            "config_id",
            "seed_gnn",
            "n_ml_seeds",
            "ml_seeds",
            "n_rows",
            "n_features",
            "accuracy_mean_std",
            "f1_macro_mean_std",
            "f1_weighted_mean_std",
            "f1_class_0_mean_std",
            "f1_class_1_mean_std",
            "recall_class_0_mean_std",
            "recall_class_1_mean_std",
            "mcc_mean_std",
            "roc_auc_mean_std",
        ]
    ].rename(
        columns={
            "graph_label": "Graph",
            "gnn_model": "GNN architecture",
            "ml_model": "ML model",
            "config_id": "Config ID",
            "seed_gnn": "GNN seed",
            "n_ml_seeds": "N ML seeds",
            "ml_seeds": "ML seeds",
            "n_rows": "N instances",
            "n_features": "N features",
            "accuracy_mean_std": "Accuracy",
            "f1_macro_mean_std": "F1-macro",
            "f1_weighted_mean_std": "F1-weighted",
            "f1_class_0_mean_std": "F1-class-0",
            "f1_class_1_mean_std": "F1-class-1",
            "recall_class_0_mean_std": "Recall class 0",
            "recall_class_1_mean_std": "Recall class 1",
            "mcc_mean_std": "MCC",
            "roc_auc_mean_std": "ROC-AUC",
        }
    )

    selected_configs = pd.DataFrame(
        [
            {
                "ml_model": model_name,
                "config_id": config["config_id"],
                "selection_rank": config["rank"],
                "selection_f1_macro_mean": config["selection_f1_macro_mean"],
                "selection_f1_macro_std": config["selection_f1_macro_std"],
                "hyperparams_json": config["hyperparams_json"],
            }
            for model_name, config in configs.items()
        ]
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_path) as writer:
        paper_table.to_excel(writer, sheet_name="paper_table", index=False)
        aggregated.to_excel(writer, sheet_name="aggregated_numeric", index=False)
        details.to_excel(writer, sheet_name="per_ml_seed_details", index=False)
        folds.to_excel(writer, sheet_name="fold_details", index=False)
        predictions.to_excel(writer, sheet_name="oof_predictions", index=False)
        selected_configs.to_excel(writer, sheet_name="selected_configs", index=False)

    _log(f"Saved: {args.output_path}")
    print(
        paper_table[
            ["Graph", "GNN architecture", "ML model", "Accuracy", "F1-macro", "F1-weighted"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Recompute selected XGBoost/LightGBM OOF metrics."
    )
    parser.add_argument("--preprocessed-dir", type=Path, default=PREPROCESSED_DIR)
    parser.add_argument("--domain-seed-path", type=Path, default=DOMAIN_SEED_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--models", default="xgboost,lightgbm")
    parser.add_argument("--ml-seeds", default="42,123,456,789,2024")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--model-jobs", type=int, default=1)
    args = parser.parse_args()
    recompute(args)


if __name__ == "__main__":
    main()


