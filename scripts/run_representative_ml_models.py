from pathlib import Path
import argparse
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml_explainer.ml_models import train_ml_models


def _log(message):
    print(f"[run_representative_ml_models] {message}")


def _resolve_path(path_value):
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_safe)


def _parse_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _parse_int_list(value):
    return [int(item) for item in _parse_list(value)]


def _load_dataset_lookup(datasets_config_path):
    config = _load_json(datasets_config_path)
    defaults = config.get("defaults", {})
    lookup = {}

    for dataset in config.get("datasets", []):
        if not dataset.get("enabled", True):
            continue
        merged = dict(defaults)
        merged.update(dataset)
        graph_name = merged.get("graph_name")
        gnn_model = merged.get("gnn_model")
        lookup[(graph_name, gnn_model)] = merged

    return lookup


def _load_representative_seeds(path):
    df = pd.read_excel(path)
    required_columns = {"graph_name", "gnn_model", "representative_seed"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Representative seed file is missing columns: {sorted(missing)}")
    return df


def _selected_rows(representatives_df, selected_graphs, selected_gnn_models):
    selected_graphs = set(selected_graphs or [])
    selected_gnn_models = set(selected_gnn_models or [])
    rows = representatives_df.copy()

    if selected_graphs:
        rows = rows[rows["graph_name"].isin(selected_graphs)]
    if selected_gnn_models:
        rows = rows[rows["gnn_model"].isin(selected_gnn_models)]

    return rows.sort_values(["graph_name", "gnn_model"])


def _contextualize(df, context):
    if df.empty:
        return []

    result = df.copy()
    for column, value in context.items():
        result[column] = value
    ordered_columns = list(context.keys()) + [
        column for column in result.columns if column not in context
    ]
    result = result[ordered_columns]
    return result.to_dict(orient="records")


def _best_param_rows(context, best_params):
    rows = []
    for model_name, tuning_result in best_params.items():
        rows.append(
            {
                **context,
                "model": model_name,
                "best_score": tuning_result.get("best_score"),
                "best_params": json.dumps(
                    tuning_result.get("best_params", {}),
                    ensure_ascii=False,
                    default=_json_safe,
                    sort_keys=True,
                ),
            }
        )
    return rows


def _best_with_test(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame()

    keys = [
        "dataset",
        "graph_name",
        "gnn_model",
        "seed_gnn",
        "seed_ml",
        "model",
    ]
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "recall_class_0",
        "f1",
        "f1_macro",
        "mcc",
        "roc_auc",
    ]
    metric_cols = [column for column in metric_cols if column in metrics_df.columns]

    val = metrics_df[metrics_df["split"] == "val"].copy()
    val = val.sort_values(
        by=["dataset", "seed_gnn", "seed_ml", "f1_macro", "mcc", "roc_auc"],
        ascending=[True, True, True, False, False, False],
    )
    best = val.groupby(["dataset", "seed_gnn", "seed_ml"], as_index=False).head(1)
    best = best.rename(columns={column: f"val_{column}" for column in metric_cols})

    test = metrics_df[metrics_df["split"] == "test"].copy()
    test = test[keys + metric_cols].rename(
        columns={column: f"test_{column}" for column in metric_cols}
    )

    return best.merge(test, on=keys, how="left")


def run_representative_ml_models(
    config_path,
    selected_graphs=None,
    selected_gnn_models=None,
    override_models=None,
    override_ml_seeds=None,
    override_n_iter=None,
    override_cv=None,
    override_n_jobs=None,
    dry_run=False,
):
    config = _load_json(config_path)
    defaults = config.get("defaults", {})

    representative_seeds_path = _resolve_path(defaults["representative_seeds_path"])
    datasets_config_path = _resolve_path(defaults["datasets_config_path"])
    output_path = _resolve_path(defaults["output_path"])
    ml_seeds = override_ml_seeds or _parse_int_list(defaults.get("ml_seeds", []))
    models = override_models or _parse_list(defaults.get("models", []))
    n_iter = int(override_n_iter if override_n_iter is not None else defaults.get("n_iter", 50))
    cv = int(override_cv if override_cv is not None else defaults.get("cv", 5))
    n_jobs = int(override_n_jobs if override_n_jobs is not None else defaults.get("n_jobs", 1))

    representatives_df = _load_representative_seeds(representative_seeds_path)
    representatives_df = _selected_rows(
        representatives_df,
        selected_graphs=selected_graphs,
        selected_gnn_models=selected_gnn_models,
    )
    dataset_lookup = _load_dataset_lookup(datasets_config_path)

    _log(f"Config: {config_path}")
    _log(f"Representative seeds: {representative_seeds_path}")
    _log(f"Datasets config: {datasets_config_path}")
    _log(f"Output: {output_path}")
    _log(f"Rows selected: {len(representatives_df)}")
    _log(f"ML seeds: {ml_seeds}")
    _log(f"Models: {models}")
    _log(f"CV: {cv} | n_iter: {n_iter} | n_jobs: {n_jobs}")
    if dry_run:
        _log("Dry run enabled: no model will be trained.")

    all_metrics = []
    all_performance = []
    all_best_params = []
    failures = []

    for _, representative in representatives_df.iterrows():
        graph_name = representative["graph_name"]
        gnn_model = representative["gnn_model"]
        seed_gnn = int(representative["representative_seed"])
        dataset_config = dataset_lookup.get((graph_name, gnn_model))

        if dataset_config is None:
            failures.append(
                {
                    "graph_name": graph_name,
                    "gnn_model": gnn_model,
                    "seed_gnn": seed_gnn,
                    "error": "No matching dataset in datasets config.",
                }
            )
            _log(f"[ERROR] No dataset for {graph_name}/{gnn_model}")
            continue

        dataset_name = dataset_config["name"]
        excel_path = _resolve_path(dataset_config["excel_path"])
        if excel_path is None or not excel_path.exists():
            failures.append(
                {
                    "dataset": dataset_name,
                    "graph_name": graph_name,
                    "gnn_model": gnn_model,
                    "seed_gnn": seed_gnn,
                    "error": f"Excel file not found: {excel_path}",
                }
            )
            _log(f"[ERROR] Missing input for {dataset_name}: {excel_path}")
            continue

        for seed_ml in ml_seeds:
            context = {
                "dataset": dataset_name,
                "graph_name": graph_name,
                "gnn_model": gnn_model,
                "seed_gnn": seed_gnn,
                "seed_ml": seed_ml,
            }
            output_dir = output_path / graph_name / gnn_model / f"seed_ml_{seed_ml}"

            _log(
                f"Dataset={dataset_name} | seed_gnn={seed_gnn} | "
                f"seed_ml={seed_ml} | models={','.join(models)}"
            )
            if dry_run:
                continue

            try:
                result = train_ml_models(
                    excel_path=str(excel_path),
                    seed_gnn=seed_gnn,
                    seed_ml=seed_ml,
                    model_names=models,
                    output_path=str(output_dir),
                    save_results=True,
                    tune=bool(defaults.get("tune", True)),
                    n_iter=n_iter,
                    cv=cv,
                    scoring=defaults.get("scoring", "f1_macro"),
                    selection_metric=defaults.get("selection_metric", "f1_macro"),
                    n_jobs=n_jobs,
                )
                all_metrics.extend(_contextualize(result["metrics"], context))
                all_performance.extend(
                    _contextualize(result["all_performance"], context)
                )
                all_best_params.extend(_best_param_rows(context, result["best_params"]))
                if result["failures"]:
                    failures.append({**context, "model_failures": result["failures"]})
            except Exception as error:
                failures.append({**context, "error": str(error)})
                _log(f"[ERROR] {dataset_name} seed_ml={seed_ml}: {error}")

    if dry_run:
        return {
            "selected_runs": len(representatives_df) * len(ml_seeds),
            "failures": failures,
        }

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    performance_df = pd.DataFrame(all_performance)
    best_params_df = pd.DataFrame(all_best_params)
    best_with_test_df = _best_with_test(metrics_df)

    metrics_path = output_path / "representative_gnn_seed_ml_metrics.xlsx"
    performance_path = output_path / "representative_gnn_seed_all_performance.xlsx"
    best_params_path = output_path / "representative_gnn_seed_best_params.xlsx"
    best_with_test_path = output_path / "representative_gnn_seed_best_with_test.xlsx"
    results_json_path = output_path / "representative_gnn_seed_ml_results.json"
    failures_path = output_path / "representative_gnn_seed_failures.json"

    _log(f"Saving metrics: {metrics_path}")
    metrics_df.to_excel(metrics_path, index=False)
    _log(f"Saving all hyperparameter performance: {performance_path}")
    performance_df.to_excel(performance_path, index=False)
    _log(f"Saving best params: {best_params_path}")
    best_params_df.to_excel(best_params_path, index=False)
    _log(f"Saving best models with test metrics: {best_with_test_path}")
    best_with_test_df.to_excel(best_with_test_path, index=False)

    _save_json(
        results_json_path,
        {
            "config": str(config_path),
            "representative_seeds_path": str(representative_seeds_path),
            "metrics_rows": len(metrics_df),
            "all_performance_rows": len(performance_df),
            "best_params_rows": len(best_params_df),
            "best_with_test_rows": len(best_with_test_df),
            "failures": failures,
        },
    )

    if failures:
        _log(f"Saving failures: {failures_path}")
        _save_json(failures_path, failures)
    elif failures_path.exists():
        failures_path.unlink()

    _log("Done")
    return {
        "metrics": metrics_df,
        "all_performance": performance_df,
        "best_params": best_params_df,
        "best_with_test": best_with_test_df,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train ML explainers on representative GNN seeds with multiple ML "
            "split seeds."
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "ml_representative_models_config.json"),
        help="Representative ML experiment config.",
    )
    parser.add_argument(
        "--graph",
        action="append",
        default=None,
        help="Run only this graph_name. Can be repeated.",
    )
    parser.add_argument(
        "--gnn-model",
        action="append",
        default=None,
        help="Run only this GNN model. Can be repeated.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Override model list, comma-separated.",
    )
    parser.add_argument(
        "--seed-ml",
        action="append",
        type=int,
        default=None,
        help="Override ML seed list. Can be repeated.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=None,
        help="Override RandomizedSearchCV iterations.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Override StratifiedKFold folds.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected runs without training.",
    )
    args = parser.parse_args()

    run_representative_ml_models(
        config_path=_resolve_path(args.config),
        selected_graphs=args.graph,
        selected_gnn_models=args.gnn_model,
        override_models=_parse_list(args.models) if args.models else None,
        override_ml_seeds=args.seed_ml,
        override_n_iter=args.n_iter,
        override_cv=args.cv,
        override_n_jobs=args.n_jobs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
