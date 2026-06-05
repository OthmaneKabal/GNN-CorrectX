from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml_explainer import add_correct_prediction_all_seeds


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_path(path_value):
    if path_value is None or str(path_value).strip() == "":
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _selected_items(items, selected_names, include_disabled):
    selected_names = set(selected_names or [])
    selected = []

    for item in items:
        if selected_names and item["name"] not in selected_names:
            continue
        if not include_disabled and not item.get("enabled", True):
            continue
        selected.append(item)

    return selected


def _validate_paths(graph_config, model_config):
    features_path = _resolve_path(graph_config["features_path"])
    artifacts_dir = _resolve_path(model_config["artifacts_dir"])

    if features_path is None or not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found for graph '{graph_config['name']}': {features_path}"
        )
    if artifacts_dir is None or not artifacts_dir.exists():
        raise FileNotFoundError(
            f"Artifacts dir not found for graph '{graph_config['name']}', "
            f"model '{model_config['name']}': {artifacts_dir}"
        )

    return features_path, artifacts_dir


def run_graph_model(graph_config, model_config, defaults):
    features_path, artifacts_dir = _validate_paths(graph_config, model_config)
    output_path = _resolve_path(model_config.get("output_path", defaults.get("output_path")))

    print(f"\n=== {graph_config['name']} / {model_config['name']} ===")
    print(f"Features: {features_path}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Output: {output_path}")

    result = add_correct_prediction_all_seeds(
        features_path=str(features_path),
        artifacts_dir=str(artifacts_dir),
        output_path=str(output_path) if output_path is not None else None,
        graph_name=graph_config["name"],
        model_name=model_config["name"],
        prediction_pattern=model_config.get(
            "prediction_pattern",
            defaults.get("prediction_pattern", "*_best_test_predictions.csv"),
        ),
        join_on=model_config.get("join_on", defaults.get("join_on", ["term"])),
        join_how=model_config.get("join_how", defaults.get("join_how", "inner")),
        feature_label_col=model_config.get("feature_label_col", defaults.get("feature_label_col")),
        prediction_label_col=model_config.get(
            "prediction_label_col",
            defaults.get("prediction_label_col"),
        ),
        prediction_col=model_config.get("prediction_col", defaults.get("prediction_col")),
    )

    return {
        "rows": len(result),
        "correct": int(result["correct_prediction"].sum()),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate all-seed correct_prediction files for configured graphs/models."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "ml_explainer_correct_prediction_config.json"),
        help="Path to the ml_explainer correct prediction config JSON.",
    )
    parser.add_argument(
        "--graph",
        action="append",
        default=None,
        help="Run only a named graph. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Run only a named model. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also run graphs/models where enabled=false.",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    defaults = config.get("defaults", {})
    graphs = _selected_items(config.get("graphs", []), args.graph, args.include_disabled)

    if not graphs:
        print("No graphs selected.")
        return

    results = {}
    failures = {}

    for graph_config in graphs:
        models = _selected_items(
            graph_config.get("models", []),
            args.model,
            args.include_disabled,
        )
        if not models:
            print(f"No models selected for graph: {graph_config['name']}")
            continue

        for model_config in models:
            key = f"{graph_config['name']}::{model_config['name']}"
            try:
                results[key] = run_graph_model(graph_config, model_config, defaults)
            except Exception as error:
                failures[key] = str(error)
                print(f"[ERROR] {key}: {error}")

    print("\n=== Summary ===")
    for key, result in results.items():
        print(
            f"{key}: rows={result['rows']}, correct={result['correct']}, "
            f"output_dir={result['output_path']}"
        )

    if failures:
        print("\nFailures:")
        for key, error in failures.items():
            print(f"{key}: {error}")


if __name__ == "__main__":
    main()
