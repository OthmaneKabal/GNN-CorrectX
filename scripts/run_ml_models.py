from pathlib import Path
import argparse
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml_explainer.ml_models import train_ml_models


def _log(message):
    print(f"[run_ml_models] {message}")


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_safe)


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _resolve_path(path_value):
    if path_value is None or str(path_value).strip() == "":
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _merge_defaults(defaults, dataset_config):
    merged = dict(defaults)
    merged.update(dataset_config)
    return merged


def _selected_datasets(datasets, selected_names, include_disabled):
    selected_names = set(selected_names or [])
    selected = []

    for dataset in datasets:
        if selected_names and dataset["name"] not in selected_names:
            continue
        if not include_disabled and not dataset.get("enabled", True):
            continue
        selected.append(dataset)

    return selected


def _parse_models(models_value):
    if isinstance(models_value, str):
        return [model.strip() for model in models_value.split(",") if model.strip()]
    return list(models_value or [])


def _parse_seed_values(seed_values):
    if seed_values is None:
        return []
    if isinstance(seed_values, int):
        return [seed_values]
    return [int(seed) for seed in seed_values]


def _infer_gnn_seeds(excel_path):
    df = pd.read_excel(excel_path, usecols=["seed"])
    seeds = sorted(int(seed) for seed in df["seed"].dropna().unique())
    return seeds


def _dataset_output_dir(base_output_path, dataset_config):
    graph_name = dataset_config.get("graph_name", "unknown_graph")
    gnn_model = dataset_config.get("gnn_model", "unknown_model")
    return Path(base_output_path) / graph_name / gnn_model


def _result_rows(dataset_config, result):
    rows = []
    metrics_df = result["metrics"].copy()
    if metrics_df.empty:
        return rows

    metrics_df.insert(0, "dataset", dataset_config["name"])
    metrics_df.insert(1, "graph_name", dataset_config.get("graph_name"))
    metrics_df.insert(2, "gnn_model", dataset_config.get("gnn_model"))
    return metrics_df.to_dict(orient="records")


def _all_performance_rows(dataset_config, result):
    performance_df = result["all_performance"].copy()
    if performance_df.empty:
        return []

    performance_df.insert(0, "dataset", dataset_config["name"])
    performance_df.insert(1, "graph_name", dataset_config.get("graph_name"))
    performance_df.insert(2, "gnn_model", dataset_config.get("gnn_model"))
    return performance_df.to_dict(orient="records")


def _best_param_rows(dataset_config, seed_gnn, seed_ml, best_params):
    rows = []
    for model_name, tuning_result in best_params.items():
        rows.append(
            {
                "dataset": dataset_config["name"],
                "graph_name": dataset_config.get("graph_name"),
                "gnn_model": dataset_config.get("gnn_model"),
                "seed_gnn": seed_gnn,
                "seed_ml": seed_ml,
                "model": model_name,
                "best_score": tuning_result.get("best_score"),
                "best_params": json.dumps(tuning_result.get("best_params", {})),
            }
        )
    return rows


def _best_model_row(dataset_config, best_model_summary):
    if not best_model_summary:
        return None

    row = dict(best_model_summary)
    row["dataset"] = dataset_config["name"]
    row["graph_name"] = dataset_config.get("graph_name")
    row["gnn_model"] = dataset_config.get("gnn_model")
    return row


def _run_one_dataset_seed(dataset_config, seed_gnn, dry_run):
    excel_path = _resolve_path(dataset_config["excel_path"])
    if excel_path is None or not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found for dataset '{dataset_config['name']}': {excel_path}"
        )

    seed_ml = int(dataset_config.get("seed_ml", 42))
    models = _parse_models(dataset_config.get("models"))
    base_output_path = _resolve_path(dataset_config.get("output_path"))
    output_dir = _dataset_output_dir(base_output_path, dataset_config)

    _log(
        f"Dataset={dataset_config['name']} | seed_gnn={seed_gnn} | "
        f"seed_ml={seed_ml} | models={','.join(models)}"
    )
    _log(f"Input: {excel_path}")
    _log(f"Output: {output_dir}")

    if dry_run:
        return None

    return train_ml_models(
        excel_path=str(excel_path),
        seed_gnn=seed_gnn,
        seed_ml=seed_ml,
        model_names=models,
        output_path=str(output_dir),
        save_results=True,
        tune=dataset_config.get("tune", False),
        n_iter=int(dataset_config.get("n_iter", 25)),
        cv=int(dataset_config.get("cv", 5)),
        scoring=dataset_config.get("scoring", "f1_macro"),
        selection_metric=dataset_config.get("selection_metric", "f1_macro"),
    )


def _seeds_for_dataset(dataset_config, forced_seeds):
    if forced_seeds:
        return forced_seeds

    configured_seeds = _parse_seed_values(dataset_config.get("gnn_seeds"))
    if configured_seeds:
        return configured_seeds

    excel_path = _resolve_path(dataset_config["excel_path"])
    return _infer_gnn_seeds(excel_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train all configured ML explainer models on all graphs and GNN seeds."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "ml_models_config.json"),
        help="Path to the ML models config JSON.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Run only a named dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seed-gnn",
        action="append",
        type=int,
        default=None,
        help="Run only a specific GNN seed. Can be passed multiple times.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Override model list with comma-separated names.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also run datasets where enabled=false.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected runs without training models.",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    defaults = config.get("defaults", {})
    datasets = [
        _merge_defaults(defaults, dataset_config)
        for dataset_config in config.get("datasets", [])
    ]
    datasets = _selected_datasets(datasets, args.dataset, args.include_disabled)

    if args.models is not None:
        for dataset in datasets:
            dataset["models"] = _parse_models(args.models)

    if not datasets:
        _log("No datasets selected.")
        return

    forced_seeds = _parse_seed_values(args.seed_gnn)
    output_path = _resolve_path(defaults.get("output_path", "outputs/ml_explainer/ml_models"))
    all_metrics = []
    all_performance = []
    all_best_params = []
    all_best_models = []
    failures = []

    _log(f"Config: {config_path}")
    _log(f"Datasets selected: {len(datasets)}")
    if args.dry_run:
        _log("Dry run enabled: no model will be trained.")

    for dataset_config in datasets:
        try:
            seeds = _seeds_for_dataset(dataset_config, forced_seeds)
        except Exception as error:
            failures.append(
                {
                    "dataset": dataset_config.get("name"),
                    "seed_gnn": None,
                    "error": str(error),
                }
            )
            _log(f"[ERROR] {dataset_config.get('name')}: {error}")
            continue

        _log(f"{dataset_config['name']} seeds: {seeds}")
        for seed_gnn in seeds:
            try:
                result = _run_one_dataset_seed(
                    dataset_config=dataset_config,
                    seed_gnn=seed_gnn,
                    dry_run=args.dry_run,
                )
                if result is None:
                    continue

                all_metrics.extend(_result_rows(dataset_config, result))
                all_performance.extend(_all_performance_rows(dataset_config, result))
                all_best_params.extend(
                    _best_param_rows(
                        dataset_config=dataset_config,
                        seed_gnn=seed_gnn,
                        seed_ml=int(dataset_config.get("seed_ml", 42)),
                        best_params=result["best_params"],
                    )
                )
                best_model_row = _best_model_row(
                    dataset_config,
                    result.get("best_model_summary"),
                )
                if best_model_row is not None:
                    all_best_models.append(best_model_row)
                if result["failures"]:
                    failures.append(
                        {
                            "dataset": dataset_config["name"],
                            "seed_gnn": seed_gnn,
                            "model_failures": result["failures"],
                        }
                    )
            except Exception as error:
                failures.append(
                    {
                        "dataset": dataset_config["name"],
                        "seed_gnn": seed_gnn,
                        "error": str(error),
                    }
                )
                _log(f"[ERROR] {dataset_config['name']} seed={seed_gnn}: {error}")

    if args.dry_run:
        return

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "all_graphs_all_seeds_ml_metrics.xlsx"
    all_performance_path = output_path / "all_performance_ml_explainer.xlsx"
    best_params_path = output_path / "all_graphs_all_seeds_best_params.xlsx"
    best_models_path = output_path / "all_graphs_all_seeds_best_models_on_val.xlsx"
    best_models_by_graph_seed_path = output_path / "best_ml_models_by_graph_seed.xlsx"
    results_json_path = output_path / "all_graphs_all_seeds_ml_results.json"
    failures_path = output_path / "all_graphs_all_seeds_failures.json"

    metrics_df = pd.DataFrame(all_metrics)
    all_performance_df = pd.DataFrame(all_performance)
    best_params_df = pd.DataFrame(all_best_params)
    best_models_df = pd.DataFrame(all_best_models)

    _log(f"Saving global metrics: {metrics_path}")
    metrics_df.to_excel(metrics_path, index=False)
    _log(f"Saving all ML explainer hyperparameter performance: {all_performance_path}")
    all_performance_df.to_excel(all_performance_path, index=False)
    _log(f"Saving global best params: {best_params_path}")
    best_params_df.to_excel(best_params_path, index=False)
    _log(f"Saving global best models: {best_models_path}")
    best_models_df.to_excel(best_models_path, index=False)
    _log(f"Saving best models by graph and seed: {best_models_by_graph_seed_path}")
    best_models_df.to_excel(best_models_by_graph_seed_path, index=False)

    _log(f"Saving global results JSON: {results_json_path}")
    _save_json(
        results_json_path,
        {
            "config": str(config_path),
            "metrics": all_metrics,
            "all_performance": all_performance,
            "best_params": all_best_params,
            "best_models_on_val": all_best_models,
            "failures": failures,
        },
    )

    if failures:
        _log(f"Saving failures: {failures_path}")
        _save_json(failures_path, failures)
    elif failures_path.exists():
        _log(f"Removing stale failures file: {failures_path}")
        failures_path.unlink()

    _log("Done")


if __name__ == "__main__":
    main()
