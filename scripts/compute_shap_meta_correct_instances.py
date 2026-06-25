"""Compute SHAP values only where the global RF meta-model is correct.

The RF is evaluated out-of-fold for each ML seed. SHAP values are computed on
the held-out fold, then only rows where rf_prediction == correct_prediction are
kept. Feature-level summaries are averaged over ML seeds.
"""

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
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
OUTPUT_DIR = ROOT / "outputs" / "reproducibility" / "shap_meta_correct_instances"
FEATURE_CATEGORIES_PATH = ROOT / "outputs" / "explanatory_features" / "biomed_cg_exp_features_categories.json"
HIGHLIGHT_FEATURES_PATH = (
    ROOT / "config" / "shap_beeswarm_highlight_features.json"
)
ML_SEEDS = [42, 123, 456, 789, 2024]
N_FOLDS = 5
DROP_FEATURES = [
    "total_degree",
    "nb_diverse_relations_total",
    "nb_is_a_total",
]
GRAPH_ORDER = [
    "biomed_cg",
    "biomed_tdg_gt2kg",
    "biomed_tdg_kggen",
    "dbpedia_174_cg",
    "dbpedia_174_tdg",
]
GNN_ORDER = ["RGCN", "RotateEGCN"]
GRAPH_LABELS = {
    "biomed_cg": "Biomed-CG",
    "biomed_tdg_gt2kg": "Biomed-TDG",
    "biomed_tdg_kggen": "Biomed-KGGen",
    "dbpedia_174_cg": "DBpedia-CG",
    "dbpedia_174_tdg": "DBpedia-TDG",
}
CATEGORY_COLORS = {
    "structural": "#4C78A8",
    "relational": "#F58518",
    "distance": "#54A24B",
    "semantic": "#B279A2",
    "other": "#4B5563",
}


def _log(message):
    print(f"[global_rf_shap_correct_meta_only] {message}")


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


def _load_feature_categories(path=FEATURE_CATEGORIES_PATH):
    with open(path, "r", encoding="utf-8") as file:
        categories = json.load(file)
    feature_to_category = {}
    for category, features in categories.items():
        for feature in features:
            feature_to_category[feature] = category
    return feature_to_category


def _load_highlight_features(path=HIGHLIGHT_FEATURES_PATH):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _underlined_text(text):
    return "".join(f"{char}\u0332" for char in text)


def _should_highlight(feature_label, rule):
    if not rule:
        return False
    if feature_label.startswith("Sum of"):
        return False

    mode = rule.get("mode", "include")
    if mode == "include":
        return feature_label in set(rule.get("features", []))
    if mode == "all_except":
        excludes = rule.get("exclude", [])
        return not any(excluded in feature_label for excluded in excludes)
    return False


def _style_y_tick_labels(ax, dataset_info, feature_to_category, highlight_rules):
    graph_label = GRAPH_LABELS.get(dataset_info["graph_name"], dataset_info["graph_name"])
    graph_gnn_label = f"{graph_label} | {dataset_info['gnn_model']}"
    rule = highlight_rules.get(graph_gnn_label)

    new_labels = []
    colors = []
    weights = []
    present_categories = []
    for label in ax.get_yticklabels():
        feature_label = label.get_text()
        category = feature_to_category.get(feature_label, "other")
        colors.append(CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"]))
        if category != "other" and category not in present_categories:
            present_categories.append(category)

        if _should_highlight(feature_label, rule):
            new_labels.append(feature_label)
            weights.append("bold")
        else:
            new_labels.append(feature_label)
            weights.append("semibold")

    ax.set_yticklabels(new_labels)
    for label, color, weight in zip(ax.get_yticklabels(), colors, weights):
        label.set_color(color)
        label.set_fontweight(weight)
    return present_categories


def _add_category_legend(fig, categories):
    handles = [
        Patch(facecolor=CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"]), label=category)
        for category in categories
    ]
    if not handles:
        return
    fig.legend(
        handles=handles,
        frameon=False,
        ncols=1,
        loc="lower left",
        bbox_to_anchor=(0.02, -0.02),
        fontsize=8,
    )


def _load_domain_seeds(path):
    df = pd.read_excel(path)
    return {row["domain"]: int(row["seed"]) for _, row in df.iterrows()}


def _load_best_rf(path):
    best = json.loads(Path(path).read_text(encoding="utf-8"))
    if best["ml_model"] != "random_forest":
        raise ValueError(f"Expected random_forest best model, got {best['ml_model']}")
    return best["config_id"], best["hyperparams"]


def _discover_datasets(preprocessed_dir, domain_seeds, dataset_filter=None, exclude_kggen=False):
    rows = []
    pattern = "*__correct_prediction_all_seeds_cleaned_ml.xlsx"
    for path in sorted(Path(preprocessed_dir).glob(pattern)):
        name = path.name.replace("__correct_prediction_all_seeds_cleaned_ml.xlsx", "")
        if "__" not in name:
            continue
        graph_name, gnn_model = name.rsplit("__", 1)
        if exclude_kggen and graph_name == "biomed_tdg_kggen":
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


def _oof_correct_meta_shap(X, y, params, seed_ml, n_folds, model_jobs):
    import shap

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_ml)
    kept_shap = []
    kept_x = []
    kept_y = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X.columns,
            index=X_test.index,
        )

        model = RandomForestClassifier(random_state=seed_ml + fold_idx, n_jobs=model_jobs)
        model.set_params(**params)
        model.set_params(random_state=seed_ml + fold_idx, n_jobs=model_jobs)
        model.fit(X_train_scaled, y_train)

        proba_correct = model.predict_proba(X_test_scaled)[:, 1]
        pred = (proba_correct >= 0.5).astype(int)
        meta_correct = pred == y_test.to_numpy()

        explainer = shap.TreeExplainer(model)
        shap_values = _positive_class_shap_values(explainer.shap_values(X_test_scaled))

        if meta_correct.any():
            kept_shap.append(shap_values[meta_correct])
            kept_x.append(X_test_scaled.iloc[meta_correct].copy())
            kept_y.append(y_test.iloc[meta_correct].copy())

        fold_predictions = pd.DataFrame(
            {
                "seed_ml": seed_ml,
                "fold": fold_idx,
                "row_index_within_seed": X_test.index,
                "correct_prediction": y_test.to_numpy(),
                "rf_oof_prediction": pred,
                "rf_oof_proba_correct": proba_correct,
                "rf_confidence": np.maximum(proba_correct, 1.0 - proba_correct),
                "meta_model_correct": meta_correct,
            }
        )
        prediction_rows.append(fold_predictions)

    if kept_shap:
        shap_correct = np.vstack(kept_shap)
        x_correct = pd.concat(kept_x, axis=0)
        y_correct = pd.concat(kept_y, axis=0)
    else:
        shap_correct = np.empty((0, X.shape[1]))
        x_correct = pd.DataFrame(columns=X.columns)
        y_correct = pd.Series(dtype=int)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    return shap_correct, x_correct, y_correct.astype(int), predictions


def _seed_feature_summary(shap_values, X_scaled, y, seed_ml):
    rows = []
    class_0_mask = y.to_numpy() == 0
    class_1_mask = y.to_numpy() == 1
    for feature_idx, feature in enumerate(X_scaled.columns):
        feature_shap = shap_values[:, feature_idx]
        feature_values = X_scaled.iloc[:, feature_idx].to_numpy()
        rows.append(
            {
                "seed_ml": seed_ml,
                "feature": feature,
                "mean_abs_shap": float(np.abs(feature_shap).mean()),
                "mean_signed_shap_all": float(feature_shap.mean()),
                "mean_signed_shap_true_class_0": float(feature_shap[class_0_mask].mean())
                if class_0_mask.any()
                else np.nan,
                "mean_signed_shap_true_class_1": float(feature_shap[class_1_mask].mean())
                if class_1_mask.any()
                else np.nan,
                "mean_abs_shap_true_class_0": float(np.abs(feature_shap[class_0_mask]).mean())
                if class_0_mask.any()
                else np.nan,
                "mean_abs_shap_true_class_1": float(np.abs(feature_shap[class_1_mask]).mean())
                if class_1_mask.any()
                else np.nan,
                "mean_feature_value_true_class_0": float(feature_values[class_0_mask].mean())
                if class_0_mask.any()
                else np.nan,
                "mean_feature_value_true_class_1": float(feature_values[class_1_mask].mean())
                if class_1_mask.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_feature_summary(seed_summaries, dataset_info, config_id, n_rows, n_features):
    combined = pd.concat(seed_summaries, ignore_index=True)
    metric_columns = [column for column in combined.columns if column not in {"seed_ml", "feature"}]
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
    agg.insert(7, "n_meta_correct_rows_mean_over_seeds", n_rows)
    agg.insert(8, "n_features", n_features)
    return agg


def _save_beeswarm(
    shap_values,
    X_scaled,
    dataset_info,
    output_dir,
    top_n,
    feature_to_category,
    highlight_rules,
):
    import shap

    if len(X_scaled) == 0:
        return
    figures_dir = output_dir / "figures" / "beeswarm_by_graph_architecture"
    figures_dir.mkdir(parents=True, exist_ok=True)
    explanation = shap.Explanation(
        values=shap_values,
        data=X_scaled.to_numpy(),
        feature_names=list(X_scaled.columns),
    )
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    fig = plt.gcf()
    ax = plt.gca()
    categories = _style_y_tick_labels(ax, dataset_info, feature_to_category, highlight_rules)
    _add_category_legend(fig, categories)
    fig.set_size_inches(8.2, 6.2)
    fig.tight_layout()
    stem = _safe_name(
        "global_rf_shap_beeswarm_correct_meta_only",
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


def _ordered_graph_gnn_labels(df):
    labels = []
    for graph_name in GRAPH_ORDER:
        for gnn_model in GNN_ORDER:
            label = f"{graph_name} | {gnn_model}"
            if ((df["graph_name"] == graph_name) & (df["gnn_model"] == gnn_model)).any():
                labels.append(label)
    return labels


def _build_heatmaps(feature_summary, output_dir, top_n):
    top_features = (
        feature_summary.groupby("feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    all_features = (
        feature_summary.groupby("feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)["feature"]
        .tolist()
    )
    df = feature_summary.copy()
    df["graph_gnn"] = df["graph_name"] + " | " + df["gnn_model"]
    for features, stem, title in [
        (
            top_features,
            f"global_rf_shap_correct_meta_only_universal_heatmap_top{top_n}",
            f"Global RF SHAP on correct meta-model predictions - Top {top_n}",
        ),
        (
            all_features,
            "global_rf_shap_correct_meta_only_all_features_heatmap",
            "Global RF SHAP on correct meta-model predictions - All features",
        ),
    ]:
        matrix = df.pivot_table(
            index="graph_gnn",
            columns="feature",
            values="mean_abs_shap",
            aggfunc="mean",
            fill_value=0.0,
        )
        matrix = matrix.reindex(
            index=_ordered_graph_gnn_labels(df),
            columns=features,
            fill_value=0.0,
        )
        table_path = output_dir / f"{stem}.xlsx"
        matrix.to_excel(table_path)
        _log(f"Saved: {table_path}")
        _save_heatmap(matrix, output_dir / "figures", stem, title)


def build_global_rf_shap_correct_meta_only(
    preprocessed_dir=PREPROCESSED_DIR,
    domain_seed_path=DOMAIN_SEED_PATH,
    best_model_path=BEST_MODEL_PATH,
    output_dir=OUTPUT_DIR,
    ml_seeds=None,
    n_folds=N_FOLDS,
    model_jobs=1,
    top_n=10,
    dataset_filter=None,
    exclude_kggen=True,
):
    ml_seeds = ml_seeds or ML_SEEDS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_seeds = _load_domain_seeds(domain_seed_path)
    config_id, params = _load_best_rf(best_model_path)
    datasets = _discover_datasets(
        preprocessed_dir,
        domain_seeds,
        dataset_filter=dataset_filter,
        exclude_kggen=exclude_kggen,
    )
    feature_to_category = _load_feature_categories()
    highlight_rules = _load_highlight_features()

    all_seed_summaries = []
    all_feature_summaries = []
    all_predictions = []
    dataset_summary_rows = []
    for dataset in datasets:
        X, y, selected_features = _load_dataset(dataset["path"], dataset["seed_gnn"])
        _log(
            f"{dataset['graph_name']}/{dataset['gnn_model']}: rows={len(X)}, "
            f"features={len(selected_features)}, seed_gnn={dataset['seed_gnn']}"
        )
        seed_summaries = []
        beeswarm_shap = []
        beeswarm_x = []
        n_correct_by_seed = []
        for seed_ml in ml_seeds:
            _log(f"  OOF SHAP seed_ml={seed_ml}")
            shap_values, X_correct, y_correct, predictions = _oof_correct_meta_shap(
                X=X,
                y=y,
                params=params,
                seed_ml=seed_ml,
                n_folds=n_folds,
                model_jobs=model_jobs,
            )
            for key in ["domain", "graph_name", "gnn_model", "dataset_name", "seed_gnn"]:
                predictions.insert(0, key, dataset[key])
            predictions.insert(6, "config_id", config_id)
            all_predictions.append(predictions)

            n_correct_by_seed.append(len(X_correct))
            if len(X_correct) == 0:
                continue
            beeswarm_shap.append(shap_values)
            beeswarm_x.append(X_correct)
            seed_summary = _seed_feature_summary(shap_values, X_correct, y_correct, seed_ml)
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

        feature_summary = _aggregate_feature_summary(
            seed_summaries=seed_summaries,
            dataset_info=dataset,
            config_id=config_id,
            n_rows=float(np.mean(n_correct_by_seed)),
            n_features=len(selected_features),
        )
        all_feature_summaries.append(feature_summary)
        if beeswarm_shap:
            _save_beeswarm(
                shap_values=np.vstack(beeswarm_shap),
                X_scaled=pd.concat(beeswarm_x, axis=0),
                dataset_info=dataset,
                output_dir=output_dir,
                top_n=top_n,
                feature_to_category=feature_to_category,
                highlight_rules=highlight_rules,
            )
        pred_all_dataset = pd.concat(
            [frame for frame in all_predictions if frame["dataset_name"].iloc[0] == dataset["dataset_name"]],
            ignore_index=True,
        )
        dataset_summary_rows.append(
            {
                **{key: dataset[key] for key in ["domain", "graph_name", "gnn_model", "dataset_name", "seed_gnn"]},
                "ml_model": "random_forest",
                "config_id": config_id,
                "n_rows_per_seed": len(X),
                "n_features": len(selected_features),
                "ml_seeds": ",".join(str(seed) for seed in ml_seeds),
                "meta_model_accuracy_mean_over_oof_rows": accuracy_score(
                    pred_all_dataset["correct_prediction"],
                    pred_all_dataset["rf_oof_prediction"],
                ),
                "meta_model_f1_macro_mean_over_oof_rows": f1_score(
                    pred_all_dataset["correct_prediction"],
                    pred_all_dataset["rf_oof_prediction"],
                    average="macro",
                    zero_division=0,
                ),
                "n_meta_correct_rows_mean_over_seeds": float(np.mean(n_correct_by_seed)),
                "n_meta_correct_rows_std_over_seeds": float(np.std(n_correct_by_seed, ddof=1)),
                "top_feature": feature_summary.iloc[0]["feature"],
                "top_feature_mean_abs_shap": feature_summary.iloc[0]["mean_abs_shap"],
            }
        )

    seed_level = pd.concat(all_seed_summaries, ignore_index=True)
    feature_level = pd.concat(all_feature_summaries, ignore_index=True)
    predictions_level = pd.concat(all_predictions, ignore_index=True)
    dataset_summary = pd.DataFrame(dataset_summary_rows)

    seed_path = output_dir / "global_rf_shap_correct_meta_only_feature_values_by_seed_ml.xlsx"
    feature_path = output_dir / "global_rf_shap_correct_meta_only_feature_values_mean_over_ml_seeds.xlsx"
    top_path = output_dir / f"global_rf_shap_correct_meta_only_top{top_n}_mean_over_ml_seeds.xlsx"
    dataset_path = output_dir / "global_rf_shap_correct_meta_only_dataset_summary.xlsx"
    prediction_path = output_dir / "global_rf_shap_correct_meta_only_oof_predictions.xlsx"
    seed_level.to_excel(seed_path, index=False)
    feature_level.to_excel(feature_path, index=False)
    feature_level[feature_level["rank"] <= top_n].to_excel(top_path, index=False)
    dataset_summary.to_excel(dataset_path, index=False)
    predictions_level.to_excel(prediction_path, index=False)
    _log(f"Saved: {seed_path}")
    _log(f"Saved: {feature_path}")
    _log(f"Saved: {top_path}")
    _log(f"Saved: {dataset_path}")
    _log(f"Saved: {prediction_path}")
    _build_heatmaps(feature_level, output_dir=output_dir, top_n=top_n)
    return feature_level


def main():
    parser = argparse.ArgumentParser(
        description="Compute global RF SHAP only on correct meta-model OOF predictions."
    )
    parser.add_argument("--preprocessed-dir", default=str(PREPROCESSED_DIR))
    parser.add_argument("--domain-seed-path", default=str(DOMAIN_SEED_PATH))
    parser.add_argument("--best-model-path", default=str(BEST_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--ml-seeds", default=",".join(str(seed) for seed in ML_SEEDS))
    parser.add_argument("--n-folds", default=N_FOLDS, type=int)
    parser.add_argument("--model-jobs", default=1, type=int)
    parser.add_argument("--top-n", default=10, type=int)
    parser.add_argument("--include-kggen", action="store_true", help="Include KGGen datasets if present.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset filter, e.g. biomed_cg or biomed_cg__RGCN. Can be repeated.",
    )
    args = parser.parse_args()
    build_global_rf_shap_correct_meta_only(
        preprocessed_dir=args.preprocessed_dir,
        domain_seed_path=args.domain_seed_path,
        best_model_path=args.best_model_path,
        output_dir=args.output_dir,
        ml_seeds=_parse_int_list(args.ml_seeds),
        n_folds=args.n_folds,
        model_jobs=args.model_jobs,
        top_n=args.top_n,
        dataset_filter=set(args.dataset) if args.dataset else None,
        exclude_kggen=not args.include_kggen,
    )


if __name__ == "__main__":
    main()



