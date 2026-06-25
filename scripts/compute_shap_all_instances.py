"""Compute SHAP values for the selected global RF averaged over ML seeds."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREPROCESSED_DIR = ROOT / "outputs" / "ml_explainer" / "preprocessed"
DOMAIN_SEED_PATH = ROOT / "outputs" / "reproducibility" / "representative_gnn_seeds.xlsx"
BEST_MODEL_PATH = (
    ROOT / "outputs" / "reproducibility" / "meta_model_hyperparameter_search"
    / "selected_global_ml_model_and_hyperparams.json"
)
OUTPUT_DIR = ROOT / "outputs" / "reproducibility" / "shap_all_instances"
ML_SEEDS = [42, 123, 456, 789, 2024]
DROP_FEATURES = [
    "total_degree",
    "nb_diverse_relations_total",
    "nb_is_a_total",
]
EXCLUDED_GRAPHS = {"biomed_tdg_kggen"}
GRAPH_ORDER = [
    "biomed_cg",
    "biomed_tdg_gt2kg",
    "biomed_tdg_kggen",
    "dbpedia_174_cg",
    "dbpedia_174_tdg",
]
GNN_ORDER = ["RGCN", "RotateEGCN"]


def _log(message):
    print(f"[global_rf_shap_mean_seeds] {message}")


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


def _safe_name(*parts):
    text = "__".join(str(part) for part in parts)
    for char in [" ", "/", "\\", ":", ",", "(", ")"]:
        text = text.replace(char, "_")
    return text


def _load_domain_seeds(path):
    df = pd.read_excel(path)
    return {row["domain"]: int(row["seed"]) for _, row in df.iterrows()}


def _load_best_rf(path):
    best = json.loads(Path(path).read_text(encoding="utf-8"))
    if best["ml_model"] != "random_forest":
        raise ValueError(f"Expected random_forest best model, got {best['ml_model']}")
    return best["config_id"], best["hyperparams"]


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


def _positive_class_shap_values(raw_values):
    if isinstance(raw_values, list) and len(raw_values) == 2:
        return np.asarray(raw_values[1])

    values = np.asarray(raw_values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            return values[:, :, 1]
        if values.shape[0] == 2:
            return values[1]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def _fit_scaled_rf_and_shap(X, y, params, seed_ml):
    import shap

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

    model = RandomForestClassifier(random_state=seed_ml, n_jobs=-1)
    model.set_params(**params)
    model.set_params(random_state=seed_ml, n_jobs=-1)
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X_scaled)
    shap_values = _positive_class_shap_values(raw_values)
    return shap_values, X_scaled


def _seed_feature_summary(shap_values, X_scaled, y, seed_ml):
    rows = []
    correct_mask = y.to_numpy() == 1
    incorrect_mask = y.to_numpy() == 0
    for feature_idx, feature in enumerate(X_scaled.columns):
        feature_shap = shap_values[:, feature_idx]
        feature_values = X_scaled.iloc[:, feature_idx].to_numpy()
        rows.append(
            {
                "seed_ml": seed_ml,
                "feature": feature,
                "mean_abs_shap": float(np.abs(feature_shap).mean()),
                "mean_signed_shap_all": float(feature_shap.mean()),
                "mean_signed_shap_correct": float(feature_shap[correct_mask].mean())
                if correct_mask.any()
                else np.nan,
                "mean_signed_shap_incorrect": float(feature_shap[incorrect_mask].mean())
                if incorrect_mask.any()
                else np.nan,
                "mean_abs_shap_correct": float(np.abs(feature_shap[correct_mask]).mean())
                if correct_mask.any()
                else np.nan,
                "mean_abs_shap_incorrect": float(np.abs(feature_shap[incorrect_mask]).mean())
                if incorrect_mask.any()
                else np.nan,
                "mean_feature_value_correct": float(feature_values[correct_mask].mean())
                if correct_mask.any()
                else np.nan,
                "mean_feature_value_incorrect": float(feature_values[incorrect_mask].mean())
                if incorrect_mask.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_feature_summary(seed_summaries, dataset_info, config_id, n_rows, n_features):
    combined = pd.concat(seed_summaries, ignore_index=True)
    metric_columns = [
        column
        for column in combined.columns
        if column not in {"seed_ml", "feature"}
    ]
    agg = combined.groupby("feature", as_index=False).agg(
        **{f"{column}_mean_over_seeds": (column, "mean") for column in metric_columns},
        **{f"{column}_std_over_seeds": (column, "std") for column in metric_columns},
    )
    agg = agg.rename(columns={"mean_abs_shap_mean_over_seeds": "mean_abs_shap"})
    agg = agg.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)
    for key in ["domain", "graph_name", "gnn_model", "dataset_name", "seed_gnn"]:
        agg.insert(0, key, dataset_info[key])
    agg.insert(5, "ml_model", "random_forest")
    agg.insert(6, "config_id", config_id)
    agg.insert(7, "n_rows", n_rows)
    agg.insert(8, "n_features", n_features)
    return agg


def _save_heatmap(matrix, output_dir, stem, title, cmap="YlGnBu"):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_width = max(12, 0.75 * matrix.shape[1] + 5)
    fig_height = max(7, 0.45 * matrix.shape[0] + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.values, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Graph | GNN")
    ax.set_xticks(range(matrix.shape[1]), labels=matrix.columns, rotation=90)
    ax.set_yticks(range(matrix.shape[0]), labels=matrix.index)

    max_value = np.nanmax(matrix.values) if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iloc[row_idx, col_idx]
            color = "black" if value < max_value * 0.55 else "white"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Mean absolute SHAP value")
    fig.tight_layout()
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    _log(f"Saved: {png_path}")
    _log(f"Saved: {pdf_path}")


def _save_beeswarm(mean_shap_values, X_scaled, dataset_info, output_dir, top_n):
    import shap

    figures_dir = output_dir / "figures" / "beeswarm_by_graph_architecture"
    figures_dir.mkdir(parents=True, exist_ok=True)

    explanation = shap.Explanation(
        values=mean_shap_values,
        data=X_scaled.to_numpy(),
        feature_names=list(X_scaled.columns),
    )
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.2, 6.2)
    plt.title(
        "Global RF SHAP distribution averaged over ML seeds - "
        f"{dataset_info['graph_name']} / {dataset_info['gnn_model']}",
        fontsize=11,
    )
    fig.tight_layout()

    stem = _safe_name(
        "global_rf_mean_seed_shap_beeswarm",
        dataset_info["graph_name"],
        dataset_info["gnn_model"],
        f"top{top_n}",
    )
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved: {png_path}")
    _log(f"Saved: {pdf_path}")


def _save_signed_class_heatmap(feature_summary, dataset_info, output_dir, top_n):
    figures_dir = output_dir / "figures" / "signed_impact_by_graph_architecture"
    selected = feature_summary.sort_values("mean_abs_shap", ascending=False).head(top_n)
    matrix = selected[
        [
            "mean_signed_shap_incorrect_mean_over_seeds",
            "mean_signed_shap_correct_mean_over_seeds",
        ]
    ].to_numpy()
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.01)

    fig_height = max(5.2, 0.42 * len(selected) + 1.7)
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(
        "Global RF SHAP signed impact averaged over ML seeds - "
        f"{dataset_info['graph_name']} / {dataset_info['gnn_model']}",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel("True GNN prediction status")
    ax.set_ylabel("Feature")
    ax.set_xticks([0, 1], labels=["Incorrect (0)", "Correct (1)"])
    ax.set_yticks(range(len(selected)), labels=selected["feature"])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if abs(value) > vmax * 0.55 else "#111827"
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(selected))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean SHAP value toward correct_prediction=1", rotation=270, labelpad=18)
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(
        "global_rf_mean_seed_shap_signed_impact",
        dataset_info["graph_name"],
        dataset_info["gnn_model"],
        f"top{top_n}",
    )
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved: {png_path}")
    _log(f"Saved: {pdf_path}")


def _ordered_graph_gnn_labels(df):
    labels = []
    for graph_name in GRAPH_ORDER:
        for gnn_model in GNN_ORDER:
            label = f"{graph_name} | {gnn_model}"
            if ((df["graph_name"] == graph_name) & (df["gnn_model"] == gnn_model)).any():
                labels.append(label)
    return labels


def _build_universal_heatmap(feature_summary, output_dir, top_n):
    top_features = (
        feature_summary.groupby("feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    df = feature_summary.copy()
    df["graph_gnn"] = df["graph_name"] + " | " + df["gnn_model"]
    matrix = df.pivot_table(
        index="graph_gnn",
        columns="feature",
        values="mean_abs_shap",
        aggfunc="mean",
        fill_value=0.0,
    )
    matrix = matrix.reindex(
        index=_ordered_graph_gnn_labels(df),
        columns=top_features,
        fill_value=0.0,
    )
    table_path = output_dir / f"global_rf_mean_seed_shap_universal_heatmap_top{top_n}.xlsx"
    matrix.to_excel(table_path)
    _log(f"Saved: {table_path}")
    _save_heatmap(
        matrix,
        output_dir=output_dir / "figures",
        stem=f"global_rf_mean_seed_shap_universal_heatmap_top{top_n}",
        title=f"Global RF SHAP importance averaged over ML seeds - Top {top_n}",
    )


def _build_all_features_heatmap(feature_summary, output_dir):
    all_features = (
        feature_summary.groupby("feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)["feature"]
        .tolist()
    )
    df = feature_summary.copy()
    df["graph_gnn"] = df["graph_name"] + " | " + df["gnn_model"]
    matrix = df.pivot_table(
        index="graph_gnn",
        columns="feature",
        values="mean_abs_shap",
        aggfunc="mean",
        fill_value=0.0,
    )
    matrix = matrix.reindex(
        index=_ordered_graph_gnn_labels(df),
        columns=all_features,
        fill_value=0.0,
    )
    table_path = output_dir / "global_rf_mean_seed_shap_all_features_heatmap.xlsx"
    matrix.to_excel(table_path)
    _log(f"Saved: {table_path}")
    _save_heatmap(
        matrix,
        output_dir=output_dir / "figures",
        stem="global_rf_mean_seed_shap_all_features_heatmap",
        title="Global RF SHAP importance averaged over ML seeds - All features",
    )


def build_global_rf_shap_mean_over_ml_seeds(
    preprocessed_dir=PREPROCESSED_DIR,
    domain_seed_path=DOMAIN_SEED_PATH,
    best_model_path=BEST_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    ml_seeds=None,
    top_n=10,
    dataset_filter=None,
):
    ml_seeds = ml_seeds or ML_SEEDS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_seeds = _load_domain_seeds(domain_seed_path)
    config_id, params = _load_best_rf(best_model_path)
    datasets = _discover_datasets(preprocessed_dir, domain_seeds, dataset_filter=dataset_filter)

    all_seed_summaries = []
    all_feature_summaries = []
    dataset_summary_rows = []
    mean_shap_npz_payload = {}
    for dataset in datasets:
        X, y, selected_features = _load_dataset(dataset["path"], dataset["seed_gnn"])
        _log(
            f"{dataset['graph_name']}/{dataset['gnn_model']}: rows={len(X)}, "
            f"features={len(selected_features)}, seed_gnn={dataset['seed_gnn']}"
        )
        seed_summaries = []
        seed_shap_values = []
        x_scaled_reference = None
        for seed_ml in ml_seeds:
            _log(f"  SHAP seed_ml={seed_ml}")
            shap_values, X_scaled = _fit_scaled_rf_and_shap(X, y, params, seed_ml)
            seed_shap_values.append(shap_values)
            if x_scaled_reference is None:
                x_scaled_reference = X_scaled
            seed_summary = _seed_feature_summary(shap_values, X_scaled, y, seed_ml)
            for key in ["domain", "graph_name", "gnn_model", "dataset_name", "seed_gnn"]:
                seed_summary.insert(0, key, dataset[key])
            seed_summary.insert(5, "ml_model", "random_forest")
            seed_summary.insert(6, "config_id", config_id)
            all_seed_summaries.append(seed_summary)
            seed_summaries.append(seed_summary.drop(columns=[
                "domain",
                "graph_name",
                "gnn_model",
                "dataset_name",
                "seed_gnn",
                "ml_model",
                "config_id",
            ]))

        mean_shap_values = np.mean(np.stack(seed_shap_values, axis=0), axis=0)
        feature_summary = _aggregate_feature_summary(
            seed_summaries=seed_summaries,
            dataset_info=dataset,
            config_id=config_id,
            n_rows=len(X),
            n_features=len(selected_features),
        )
        all_feature_summaries.append(feature_summary)
        mean_shap_npz_payload[f"{dataset['graph_name']}__{dataset['gnn_model']}__values"] = mean_shap_values
        mean_shap_npz_payload[f"{dataset['graph_name']}__{dataset['gnn_model']}__x_scaled"] = (
            x_scaled_reference.to_numpy()
        )
        mean_shap_npz_payload[f"{dataset['graph_name']}__{dataset['gnn_model']}__y"] = y.to_numpy()
        mean_shap_npz_payload[f"{dataset['graph_name']}__{dataset['gnn_model']}__features"] = np.asarray(
            list(x_scaled_reference.columns)
        )
        _save_beeswarm(
            mean_shap_values=mean_shap_values,
            X_scaled=x_scaled_reference,
            dataset_info=dataset,
            output_dir=output_dir,
            top_n=top_n,
        )
        _save_signed_class_heatmap(
            feature_summary=feature_summary,
            dataset_info=dataset,
            output_dir=output_dir,
            top_n=top_n,
        )
        dataset_summary_rows.append(
            {
                **{key: dataset[key] for key in ["domain", "graph_name", "gnn_model", "dataset_name", "seed_gnn"]},
                "ml_model": "random_forest",
                "config_id": config_id,
                "n_rows": len(X),
                "n_features": len(selected_features),
                "ml_seeds": ",".join(str(seed) for seed in ml_seeds),
                "top_feature": feature_summary.iloc[0]["feature"],
                "top_feature_mean_abs_shap": feature_summary.iloc[0]["mean_abs_shap"],
            }
        )

    seed_level = pd.concat(all_seed_summaries, ignore_index=True)
    feature_level = pd.concat(all_feature_summaries, ignore_index=True)
    dataset_summary = pd.DataFrame(dataset_summary_rows)

    seed_path = output_dir / "global_rf_shap_feature_values_by_seed_ml.xlsx"
    feature_path = output_dir / "global_rf_shap_feature_values_mean_over_ml_seeds.xlsx"
    top_path = output_dir / f"global_rf_shap_top{top_n}_mean_over_ml_seeds.xlsx"
    dataset_path = output_dir / "global_rf_shap_dataset_summary.xlsx"
    mean_shap_path = output_dir / "global_rf_mean_seed_sample_shap_values.npz"
    seed_level.to_excel(seed_path, index=False)
    feature_level.to_excel(feature_path, index=False)
    feature_level[feature_level["rank"] <= top_n].to_excel(top_path, index=False)
    dataset_summary.to_excel(dataset_path, index=False)
    np.savez_compressed(mean_shap_path, **mean_shap_npz_payload)
    _log(f"Saved: {seed_path}")
    _log(f"Saved: {feature_path}")
    _log(f"Saved: {top_path}")
    _log(f"Saved: {dataset_path}")
    _log(f"Saved: {mean_shap_path}")

    _build_universal_heatmap(feature_level, output_dir=output_dir, top_n=top_n)
    _build_all_features_heatmap(feature_level, output_dir=output_dir)
    return feature_level


def main():
    parser = argparse.ArgumentParser(
        description="Compute global RF SHAP values averaged over ML seeds."
    )
    parser.add_argument("--preprocessed-dir", default=str(PREPROCESSED_DIR))
    parser.add_argument("--domain-seed-path", default=str(DOMAIN_SEED_PATH))
    parser.add_argument("--best-model-path", default=str(BEST_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--ml-seeds", default=",".join(str(seed) for seed in ML_SEEDS))
    parser.add_argument("--top-n", default=10, type=int)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset filter, e.g. biomed_cg or biomed_cg__RGCN. Can be repeated.",
    )
    args = parser.parse_args()
    build_global_rf_shap_mean_over_ml_seeds(
        preprocessed_dir=args.preprocessed_dir,
        domain_seed_path=args.domain_seed_path,
        best_model_path=args.best_model_path,
        output_dir=args.output_dir,
        ml_seeds=_parse_int_list(args.ml_seeds),
        top_n=args.top_n,
        dataset_filter=set(args.dataset) if args.dataset else None,
    )


if __name__ == "__main__":
    main()




