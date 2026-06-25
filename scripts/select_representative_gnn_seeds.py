"""Select one representative GNN seed per domain.

The seed is selected with the same principle as the previous representative
seed script: distance to the mean GNN performance. Here the distances are
aggregated across all graph/architecture runs in a domain, so the selected
seed is common to the whole domain.
"""

from pathlib import Path
import argparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reproducibility"
DEFAULT_METRICS = ["test_acc", "test_f1"]

KG_NAME_TO_GRAPH = {
    "reference_kg": "biomed_cg",
    "GT2KG_kg": "biomed_tdg_gt2kg",
        "DBpedia_174_clean_kg": "dbpedia_174_cg",
    "dbpedia_174_noisy_final_KG": "dbpedia_174_tdg",
}


def _log(message):
    print(f"[select_representative_gnn_seeds] {message}")


def _resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _parse_list(text):
    if text is None or text.strip() == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_model_name(model_name):
    if "RotatEGCN" in model_name or "RotateEGCN" in model_name:
        return "RotateEGCN"
    return model_name


def _infer_domain(path):
    parts = [part.lower() for part in path.parts]
    if "biomedical" in parts:
        return "biomedical"
    if "dbpedia_174" in parts or "dbpedia" in str(path).lower():
        return "dbpedia"
    return "unknown"


def _infer_kg_name(per_seed_path):
    name = per_seed_path.stem
    if name.startswith("per_seed_"):
        name = name[len("per_seed_") :]
    parts = name.split("__")
    return parts[0] if parts else None


def _infer_model_name(per_seed_path):
    name = per_seed_path.stem
    parts = name.split("__")
    if len(parts) >= 3:
        return _normalize_model_name(parts[2])
    return "unknown"


def _read_per_seed(path):
    df = pd.read_csv(path)
    if "seed" not in df.columns:
        raise ValueError(f"{path} must contain a 'seed' column.")
    return df


def _file_seed_distances(path, metrics):
    df = _read_per_seed(path)
    missing_metrics = [metric for metric in metrics if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"{path} is missing metrics: {missing_metrics}")

    score_df = df[metrics].astype(float)
    means = score_df.mean()
    distances = ((score_df - means) ** 2).sum(axis=1) ** 0.5

    kg_name = _infer_kg_name(path)
    graph_name = KG_NAME_TO_GRAPH.get(kg_name, kg_name)
    gnn_model = _infer_model_name(path)
    domain = _infer_domain(path)

    rows = []
    for index, row in df.iterrows():
        result = {
            "domain": domain,
            "graph_name": graph_name,
            "kg_name": kg_name,
            "gnn_model": gnn_model,
            "seed": int(row["seed"]),
            "distance_to_file_mean": float(distances.iloc[index]),
            "per_seed_file": str(path),
            "metrics_used": ",".join(metrics),
        }
        for metric in metrics:
            metric_value = float(row[metric])
            result[f"{metric}_mean_for_file"] = float(means[metric])
            result[f"{metric}_value"] = metric_value
            result[f"{metric}_abs_diff"] = abs(metric_value - float(means[metric]))
        rows.append(result)
    return rows


def select_domain_representative_gnn_seed(
    input_root=DEFAULT_INPUT_ROOT,
    output_dir=DEFAULT_OUTPUT_DIR,
    metrics=None,
    include_patterns=None,
    exclude_patterns=None,
):
    input_root = _resolve_path(input_root)
    output_dir = _resolve_path(output_dir)
    metrics = metrics or DEFAULT_METRICS
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []

    files = sorted(input_root.rglob("per_seed_*.csv"))
    if include_patterns:
        files = [
            path
            for path in files
            if any(pattern.lower() in str(path).lower() for pattern in include_patterns)
        ]
    if exclude_patterns:
        files = [
            path
            for path in files
            if not any(pattern.lower() in str(path).lower() for pattern in exclude_patterns)
        ]
    if not files:
        raise FileNotFoundError(f"No per_seed_*.csv files found in {input_root}")

    _log(f"Files found: {len(files)}")
    _log(f"Metrics used: {metrics}")

    detail_rows = []
    for path in files:
        _log(f"Reading: {path}")
        detail_rows.extend(_file_seed_distances(path, metrics))

    details = pd.DataFrame(detail_rows)
    details = details[details["domain"] != "unknown"].copy()
    if details.empty:
        raise ValueError("No domain could be inferred from the per_seed files.")

    metric_abs_diff_columns = [f"{metric}_abs_diff" for metric in metrics]
    grouped = details.groupby(["domain", "seed"], as_index=False).agg(
        aggregate_distance=("distance_to_file_mean", "mean"),
        aggregate_distance_std=("distance_to_file_mean", "std"),
        aggregate_distance_sum=("distance_to_file_mean", "sum"),
        n_graph_architecture_runs=("per_seed_file", "nunique"),
        graphs=("graph_name", lambda values: ",".join(sorted(set(values)))),
        gnn_models=("gnn_model", lambda values: ",".join(sorted(set(values)))),
        **{
            f"{column}_mean": (column, "mean")
            for column in metric_abs_diff_columns
        },
    )

    selected_rows = []
    rankings = []
    for domain, domain_df in grouped.groupby("domain"):
        ranked = domain_df.sort_values(
            by=["aggregate_distance", "aggregate_distance_std", "seed"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        rankings.append(ranked)
        selected_rows.append(ranked.iloc[0].to_dict())

    selected = pd.DataFrame(selected_rows).sort_values("domain").reset_index(drop=True)
    ranking = pd.concat(rankings, ignore_index=True).sort_values(["domain", "rank"])
    details = details.sort_values(["domain", "graph_name", "gnn_model", "seed"])

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "representative_gnn_seeds.xlsx"
    ranking_path = output_dir / "representative_gnn_seed_ranking.xlsx"
    details_path = output_dir / "representative_gnn_seed_details.xlsx"
    selected.to_excel(selected_path, index=False)
    ranking.to_excel(ranking_path, index=False)
    details.to_excel(details_path, index=False)

    _log(f"Saved: {selected_path}")
    _log(f"Saved: {ranking_path}")
    _log(f"Saved: {details_path}")
    print(selected.to_string(index=False))
    return selected, ranking, details


def main():
    parser = argparse.ArgumentParser(
        description="Select one representative GNN seed per domain."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics used to compute distance to mean.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="Keep only files whose path contains this pattern. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Drop files whose path contains this pattern. Can be repeated.",
    )
    args = parser.parse_args()

    select_domain_representative_gnn_seed(
        input_root=args.input_root,
        output_dir=args.output_dir,
        metrics=_parse_list(args.metrics),
        include_patterns=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
    )


if __name__ == "__main__":
    main()

