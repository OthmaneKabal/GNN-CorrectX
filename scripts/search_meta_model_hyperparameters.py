"""Global hyperparameter search with fixed OOF evaluation.

Goal:
    Select one global ML model family and one global hyperparameter set.

Protocol:
    For each model family and sampled hyperparameter configuration:
      1. Evaluate the same hyperparameters on every graph/GNN dataset.
      2. Repeat OOF evaluation for multiple ML random states.
      3. Aggregate all OOF scores.
      4. Rank configurations with mean(F1-macro) - std(F1-macro).

Unlike nested CV, hyperparameters do not change per test fold. This makes the
selected model/configuration easier to report and reuse across all graphes.
"""

from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREPROCESSED_DIR = ROOT / "outputs" / "ml_explainer" / "preprocessed"
DOMAIN_SEED_PATH = ROOT / "outputs" / "reproducibility" / "representative_gnn_seeds.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "reproducibility" / "meta_model_hyperparameter_search"

DEFAULT_MODELS = ["random_forest", "xgboost", "lightgbm"]
DEFAULT_ML_SEEDS = [42, 123, 456, 789, 2024]
DROP_FEATURES = [
    "total_degree",
    "nb_diverse_relations_total",
    "nb_is_a_total",
]
EXCLUDED_GRAPHS = {"biomed_tdg_kggen"}


def _log(message):
    print(f"[meta_model_hyperparameter_search] {message}", flush=True)


def _parse_list(text):
    if text is None or str(text).strip() == "":
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_int_list(text):
    return [int(item) for item in _parse_list(text)]


def _domain_from_graph(graph_name):
    if str(graph_name).startswith("biomed"):
        return "biomedical"
    if str(graph_name).startswith("dbpedia"):
        return "dbpedia"
    raise ValueError(f"Cannot infer domain from graph name: {graph_name}")


def _json_dumps(value):
    return json.dumps(value, sort_keys=True)


def _config_id(model_name, params):
    text = f"{model_name}:{_json_dumps(params)}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _load_domain_seeds(path):
    df = pd.read_excel(path)
    required = {"domain", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Domain seed file missing columns: {sorted(missing)}")
    return {row["domain"]: int(row["seed"]) for _, row in df.iterrows()}


def _discover_datasets(preprocessed_dir, domain_seeds, dataset_filter=None):
    rows = []
    for path in sorted(Path(preprocessed_dir).glob("*__correct_prediction_all_seeds_cleaned_ml.xlsx")):
        name = path.name.replace("__correct_prediction_all_seeds_cleaned_ml.xlsx", "")
        if "__" not in name:
            continue
        graph_name, gnn_model = name.rsplit("__", 1)
        if graph_name in EXCLUDED_GRAPHS:
            continue
        dataset_name = f"{graph_name}__{gnn_model}"
        if dataset_filter and dataset_name not in dataset_filter and graph_name not in dataset_filter:
            continue
        domain = _domain_from_graph(graph_name)
        if domain not in domain_seeds:
            raise ValueError(f"No representative seed found for domain: {domain}")
        rows.append(
            {
                "domain": domain,
                "graph_name": graph_name,
                "gnn_model": gnn_model,
                "dataset_name": dataset_name,
                "seed_gnn": domain_seeds[domain],
                "path": path,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No matching cleaned ML datasets in {preprocessed_dir}")
    return rows


def _load_dataset(path, seed_gnn):
    df = pd.read_excel(path)
    df = df[df["seed"] == seed_gnn].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for seed_gnn={seed_gnn}: {path}")

    all_features = [
        column
        for column in df.columns
        if column not in {"correct_prediction", "seed"}
    ]
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


def _base_estimator(model_name, seed, y_train, model_jobs):
    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=model_jobs,
            class_weight="balanced",
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("Install xgboost on the server: pip install xgboost") from error

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=model_jobs,
            scale_pos_weight=_scale_pos_weight(y_train),
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError("Install lightgbm on the server: pip install lightgbm") from error

        return LGBMClassifier(
            random_state=seed,
            n_jobs=model_jobs,
            verbose=-1,
            class_weight="balanced",
        )

    raise ValueError(f"Unknown model: {model_name}")


def _param_space(model_name):
    if model_name == "random_forest":
        return {
            "n_estimators": [200, 300, 500, 800, 1000],
            "max_depth": [3, 5, 8, 10, 12, None],
            "min_samples_split": [2, 5, 10, 20, 40],
            "min_samples_leaf": [1, 2, 5, 10, 20, 30],
            "max_features": ["sqrt", "log2", 0.4, 0.6, 0.8],
            "bootstrap": [True],
            "max_samples": [0.5, 0.7, 0.85, 1.0],
            "class_weight": ["balanced", "balanced_subsample"],
            "ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
        }

    if model_name == "xgboost":
        return {
            "n_estimators": [100, 200, 300, 500, 800],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.55, 0.7, 0.85, 1.0],
            "colsample_bytree": [0.55, 0.7, 0.85, 1.0],
            "min_child_weight": [1, 3, 5, 10, 20],
            "gamma": [0.0, 0.1, 0.5, 1.0, 2.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0, 5.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        }

    if model_name == "lightgbm":
        return {
            "n_estimators": [100, 200, 300, 500, 800],
            "num_leaves": [7, 15, 31, 63, 127],
            "max_depth": [-1, 2, 3, 5, 8, 12],
            "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.55, 0.7, 0.85, 1.0],
            "colsample_bytree": [0.55, 0.7, 0.85, 1.0],
            "min_child_samples": [5, 10, 20, 30, 50],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0, 5.0],
            "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
        }

    raise ValueError(f"Unknown model: {model_name}")


def _sample_configs(model_name, n_configs, seed):
    sampled = list(
        ParameterSampler(
            _param_space(model_name),
            n_iter=n_configs,
            random_state=seed,
        )
    )
    rows = []
    seen = set()
    for params in sampled:
        config_id = _config_id(model_name, params)
        if config_id in seen:
            continue
        seen.add(config_id)
        rows.append(
            {
                "ml_model": model_name,
                "config_id": config_id,
                "hyperparams": params,
                "hyperparams_json": _json_dumps(params),
            }
        )
    return rows


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
        "recall_class_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if len(set(y_true)) > 1 and not np.isnan(y_score).all():
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def _run_oof_fixed_config(X, y, model_name, params, seed_ml, n_folds, model_jobs):
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

        model = _base_estimator(
            model_name=model_name,
            seed=seed_ml + fold_idx,
            y_train=y_train,
            model_jobs=model_jobs,
        )
        model.set_params(**params)
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

    oof_metrics = _evaluate(y, oof_pred.astype(int), oof_score.astype(float))
    return oof_metrics, pd.DataFrame(fold_rows), oof_pred.astype(int), oof_score.astype(float)


def _append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _read_completed(details_path):
    if not Path(details_path).exists():
        return set()
    df = pd.read_csv(details_path)
    if df.empty:
        return set()
    return {
        (
            row["graph_name"],
            row["gnn_model"],
            row["ml_model"],
            row["config_id"],
            int(row["seed_ml"]),
        )
        for _, row in df.iterrows()
    }


def _safe_literal_eval(text):
    if isinstance(text, dict):
        return text
    return ast.literal_eval(str(text))


def _write_summary(output_dir):
    output_dir = Path(output_dir)
    details_path = output_dir / "global_hyperparam_oof_details.csv"
    summary_path = output_dir / "global_hyperparam_oof_summary.xlsx"
    best_path = output_dir / "selected_global_ml_model_and_hyperparams.json"
    if not details_path.exists():
        return None

    details = pd.read_csv(details_path)
    if details.empty:
        return None

    summary_rows = []
    group_cols = ["ml_model", "config_id", "hyperparams_json"]
    for (ml_model, config_id, hyperparams_json), group in details.groupby(group_cols):
        f1_mean = group["f1_macro"].mean()
        f1_std = group["f1_macro"].std()
        summary_rows.append(
            {
                "ml_model": ml_model,
                "config_id": config_id,
                "hyperparams_json": hyperparams_json,
                "n_evaluations": len(group),
                "n_graph_gnn_pairs": group[["graph_name", "gnn_model"]].drop_duplicates().shape[0],
                "n_ml_seeds": group["seed_ml"].nunique(),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(),
                "f1_macro_mean": f1_mean,
                "f1_macro_std": f1_std,
                "f1_macro_mean_minus_std": f1_mean - f1_std,
                "recall_class_0_mean": group["recall_class_0"].mean(),
                "mcc_mean": group["mcc"].mean(),
                "roc_auc_mean": group["roc_auc"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["f1_macro_mean_minus_std", "f1_macro_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    summary.to_excel(summary_path, index=False)

    best = summary.iloc[0].to_dict()
    best["hyperparams"] = json.loads(best["hyperparams_json"])
    Path(best_path).write_text(json.dumps(best, indent=2), encoding="utf-8")
    _log(f"Saved summary: {summary_path}")
    _log(f"Saved best config: {best_path}")
    return summary


def run_global_hyperparam_oof_search(
    preprocessed_dir=PREPROCESSED_DIR,
    domain_seed_path=DOMAIN_SEED_PATH,
    output_dir=OUTPUT_DIR,
    models=None,
    ml_seeds=None,
    n_folds=5,
    n_configs=15,
    config_seed=42,
    model_jobs=8,
    dataset_filter=None,
    save_predictions=False,
    resume=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = models or DEFAULT_MODELS
    ml_seeds = ml_seeds or DEFAULT_ML_SEEDS

    domain_seeds = _load_domain_seeds(domain_seed_path)
    datasets = _discover_datasets(preprocessed_dir, domain_seeds, dataset_filter=dataset_filter)

    config_rows = []
    for model_name in models:
        config_rows.extend(_sample_configs(model_name, n_configs, config_seed))
    configs = pd.DataFrame(config_rows)
    configs_path = output_dir / "global_hyperparam_configs.xlsx"
    configs.to_excel(configs_path, index=False)
    _log(f"Configs saved: {configs_path}")

    feature_rows = []
    loaded_datasets = {}
    for dataset in datasets:
        X, y, selected_features = _load_dataset(dataset["path"], dataset["seed_gnn"])
        loaded_datasets[dataset["dataset_name"]] = (dataset, X, y, selected_features)
        feature_rows.append(
            {
                **{key: dataset[key] for key in ["domain", "graph_name", "gnn_model", "dataset_name", "seed_gnn"]},
                "n_rows": len(X),
                "n_features": len(selected_features),
                "dropped_features": ",".join(DROP_FEATURES),
                "selected_features": ",".join(selected_features),
            }
        )
    pd.DataFrame(feature_rows).to_excel(output_dir / "global_hyperparam_oof_used_features.xlsx", index=False)

    details_path = output_dir / "global_hyperparam_oof_details.csv"
    folds_path = output_dir / "global_hyperparam_oof_fold_details.csv"
    predictions_path = output_dir / "global_hyperparam_oof_predictions.csv"
    failures_path = output_dir / "global_hyperparam_oof_failures.json"
    completed = _read_completed(details_path) if resume else set()
    failures = []
    if failures_path.exists():
        failures = json.loads(failures_path.read_text(encoding="utf-8"))

    total_jobs = len(configs) * len(loaded_datasets) * len(ml_seeds)
    job_idx = 0
    for _, config in configs.iterrows():
        model_name = config["ml_model"]
        config_id = config["config_id"]
        params = _safe_literal_eval(config["hyperparams"])
        hyperparams_json = config["hyperparams_json"]
        for dataset_name, (dataset, X, y, selected_features) in loaded_datasets.items():
            for seed_ml in ml_seeds:
                job_idx += 1
                key = (dataset["graph_name"], dataset["gnn_model"], model_name, config_id, seed_ml)
                if key in completed:
                    _log(f"Skipping completed {job_idx}/{total_jobs}: {key}")
                    continue

                _log(
                    f"Job {job_idx}/{total_jobs}: model={model_name}, config={config_id}, "
                    f"graph={dataset['graph_name']}, gnn={dataset['gnn_model']}, "
                    f"seed_gnn={dataset['seed_gnn']}, seed_ml={seed_ml}, folds={n_folds}, "
                    f"rows={len(X)}, model_jobs={model_jobs}"
                )
                start_time = time.time()
                try:
                    metrics, fold_df, oof_pred, oof_score = _run_oof_fixed_config(
                        X=X,
                        y=y,
                        model_name=model_name,
                        params=params,
                        seed_ml=seed_ml,
                        n_folds=n_folds,
                        model_jobs=model_jobs,
                    )
                    elapsed = time.time() - start_time
                    detail_row = {
                        "domain": dataset["domain"],
                        "graph_name": dataset["graph_name"],
                        "gnn_model": dataset["gnn_model"],
                        "dataset_name": dataset_name,
                        "ml_model": model_name,
                        "config_id": config_id,
                        "hyperparams_json": hyperparams_json,
                        "seed_gnn": dataset["seed_gnn"],
                        "seed_ml": seed_ml,
                        "n_folds": n_folds,
                        "n_rows": len(X),
                        "n_features": len(selected_features),
                        "elapsed_seconds": elapsed,
                        **metrics,
                    }
                    _append_csv(details_path, [detail_row])

                    fold_df.insert(0, "seed_ml", seed_ml)
                    fold_df.insert(0, "seed_gnn", dataset["seed_gnn"])
                    fold_df.insert(0, "config_id", config_id)
                    fold_df.insert(0, "ml_model", model_name)
                    fold_df.insert(0, "gnn_model", dataset["gnn_model"])
                    fold_df.insert(0, "graph_name", dataset["graph_name"])
                    fold_df.insert(0, "domain", dataset["domain"])
                    _append_csv(folds_path, fold_df.to_dict("records"))

                    if save_predictions:
                        prediction_df = pd.DataFrame(
                            {
                                "domain": dataset["domain"],
                                "graph_name": dataset["graph_name"],
                                "gnn_model": dataset["gnn_model"],
                                "ml_model": model_name,
                                "config_id": config_id,
                                "seed_gnn": dataset["seed_gnn"],
                                "seed_ml": seed_ml,
                                "row_index_within_seed": X.index,
                                "correct_prediction": y.values,
                                "oof_prediction": oof_pred.values,
                                "oof_proba_correct": oof_score.values,
                            }
                        )
                        _append_csv(predictions_path, prediction_df.to_dict("records"))
                    completed.add(key)
                except Exception as error:
                    failure = {
                        "domain": dataset["domain"],
                        "graph_name": dataset["graph_name"],
                        "gnn_model": dataset["gnn_model"],
                        "ml_model": model_name,
                        "config_id": config_id,
                        "hyperparams_json": hyperparams_json,
                        "seed_gnn": dataset["seed_gnn"],
                        "seed_ml": seed_ml,
                        "error": str(error),
                    }
                    failures.append(failure)
                    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
                    _log(f"[ERROR] {failure}")

                _write_summary(output_dir)

    summary = _write_summary(output_dir)
    if summary is not None:
        print(summary.head(20).to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Search global fixed hyperparameters with OOF evaluation."
    )
    parser.add_argument("--preprocessed-dir", default=str(PREPROCESSED_DIR))
    parser.add_argument("--domain-seed-path", default=str(DOMAIN_SEED_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--ml-seeds", default=",".join(str(seed) for seed in DEFAULT_ML_SEEDS))
    parser.add_argument("--n-folds", default=5, type=int)
    parser.add_argument("--n-configs", default=15, type=int)
    parser.add_argument("--config-seed", default=42, type=int)
    parser.add_argument(
        "--model-jobs",
        default=8,
        type=int,
        help="Threads used inside each estimator fit. Use 8-16 on large servers.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset filter, e.g. biomed_cg or biomed_cg__RGCN. Can be repeated.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_global_hyperparam_oof_search(
        preprocessed_dir=args.preprocessed_dir,
        domain_seed_path=args.domain_seed_path,
        output_dir=args.output_dir,
        models=_parse_list(args.models),
        ml_seeds=_parse_int_list(args.ml_seeds),
        n_folds=args.n_folds,
        n_configs=args.n_configs,
        config_seed=args.config_seed,
        model_jobs=args.model_jobs,
        dataset_filter=set(args.dataset) if args.dataset else None,
        save_predictions=args.save_predictions,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()




