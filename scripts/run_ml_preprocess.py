from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml_explainer import preprocess_correct_prediction_dataset


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


def _validate_dataset(dataset_config):
    input_path = _resolve_path(dataset_config["input_path"])
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found for dataset '{dataset_config['name']}': {input_path}"
        )
    return input_path


def run_dataset(dataset_config):
    input_path = _validate_dataset(dataset_config)
    output_path = _resolve_path(dataset_config.get("output_path"))

    print(f"\n=== Cleaning {dataset_config['name']} ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Add unreachable flags: {dataset_config.get('add_unreachable_flags', False)}")

    result = preprocess_correct_prediction_dataset(
        input_path=str(input_path),
        output_path=str(output_path) if output_path is not None else None,
        add_flags=dataset_config.get("add_unreachable_flags", False),
        drop_columns=dataset_config.get("drop_columns"),
        distance_columns=dataset_config.get("distance_columns"),
    )

    return {
        "rows": len(result),
        "columns": len(result.columns),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean configured all-seeds correct-prediction datasets for ML."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "ml_preprocess_config.json"),
        help="Path to the ML preprocessing config JSON.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Run only a named dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also run datasets where enabled=false.",
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

    if not datasets:
        print("No datasets selected.")
        return

    results = {}
    failures = {}
    for dataset_config in datasets:
        try:
            results[dataset_config["name"]] = run_dataset(dataset_config)
        except Exception as error:
            failures[dataset_config["name"]] = str(error)
            print(f"[ERROR] {dataset_config['name']}: {error}")

    print("\n=== Summary ===")
    for dataset_name, result in results.items():
        print(
            f"{dataset_name}: rows={result['rows']}, columns={result['columns']}, "
            f"output_dir={result['output_path']}"
        )

    if failures:
        print("\nFailures:")
        for dataset_name, error in failures.items():
            print(f"{dataset_name}: {error}")


if __name__ == "__main__":
    main()
