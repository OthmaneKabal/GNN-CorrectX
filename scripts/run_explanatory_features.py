from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explanatory_features_extraction import ExplanatoryFeaturesExtraction


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


def _merge_defaults(defaults, graph_config):
    merged = dict(defaults)
    merged.update(graph_config)
    return merged


def _selected_graphs(graphs, selected_names, include_disabled):
    selected = []
    selected_names = set(selected_names or [])

    for graph in graphs:
        if selected_names and graph["name"] not in selected_names:
            continue
        if not include_disabled and not graph.get("enabled", True):
            continue
        selected.append(graph)

    return selected


def _validate_graph_config(graph_config):
    graph_path = _resolve_path(graph_config.get("graph_path"))
    gs_path = _resolve_path(graph_config.get("gs_path"))

    if graph_path is None or not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if gs_path is None or not gs_path.exists():
        raise FileNotFoundError(
            f"Gold-standard file not found for '{graph_config['name']}': {gs_path}"
        )

    return graph_path, gs_path


def run_graph(graph_config):
    graph_path, gs_path = _validate_graph_config(graph_config)
    output_path = _resolve_path(graph_config.get("output_path"))

    print(f"\n=== Running {graph_config['name']} ===")
    print(f"Graph: {graph_path}")
    print(f"GS: {gs_path}")
    print(f"Features: {graph_config['features']}")
    print(f"Relations: {graph_config.get('relations_to_count', [])}")
    print(f"Output: {output_path}")

    extractor = ExplanatoryFeaturesExtraction(
        graph_path=str(graph_path),
        gs_path=str(gs_path),
        features=graph_config["features"],
        relations_to_count=graph_config.get("relations_to_count", []),
        path_cutoff=graph_config.get("path_cutoff", 4),
        plm=graph_config["plm"],
        output_path=str(output_path) if output_path is not None else None,
    )
    return extractor.extract(save=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run explanatory feature extraction for graphs listed in config."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "explanatory_features_config.json"),
        help="Path to the explanatory feature extraction config JSON.",
    )
    parser.add_argument(
        "--graph",
        action="append",
        default=None,
        help="Run only a named graph. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Also run graphs where enabled=false.",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    defaults = config.get("defaults", {})
    graphs = [
        _merge_defaults(defaults, graph_config)
        for graph_config in config.get("graphs", [])
    ]
    graphs = _selected_graphs(graphs, args.graph, args.include_disabled)

    if not graphs:
        print("No graphs selected.")
        return

    results = {}
    failures = {}
    for graph_config in graphs:
        try:
            result = run_graph(graph_config)
            results[graph_config["name"]] = {
                "excel_path": str(result["excel_path"]),
                "json_path": str(result["json_path"]),
                "rows": len(result["features"]),
                "columns": len(result["features"].columns),
            }
        except Exception as error:
            failures[graph_config["name"]] = str(error)
            print(f"[ERROR] {graph_config['name']}: {error}")

    print("\n=== Summary ===")
    for graph_name, result in results.items():
        print(
            f"{graph_name}: rows={result['rows']}, columns={result['columns']}, "
            f"excel={result['excel_path']}, json={result['json_path']}"
        )

    if failures:
        print("\nFailures:")
        for graph_name, error in failures.items():
            print(f"{graph_name}: {error}")


if __name__ == "__main__":
    main()
