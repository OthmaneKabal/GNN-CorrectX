"""Add model correctness labels to explanatory feature files."""

from pathlib import Path
import argparse
import re

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_explainer"


def _log(message):
    print(f"[ml_explainer] {message}")


def _read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _find_column(df, candidates, role):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find {role} column. Tried {candidates}. "
        f"Available columns: {df.columns.tolist()}"
    )


def _normalize_text(series):
    return series.astype(str).str.strip()


def _resolve_output_path(features_path, predictions_path, output_path):
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix:
            return output_path
        return output_path / f"{Path(features_path).stem}_with_correct_prediction.xlsx"

    return (
        DEFAULT_OUTPUT_DIR
        / f"{Path(features_path).stem}__{Path(predictions_path).stem}_with_correct_prediction.xlsx"
    )


def _add_correct_prediction_to_frames(
    features,
    predictions,
    join_on,
    join_how,
    feature_label_col=None,
    prediction_label_col=None,
    prediction_col=None,
):
    feature_label_col = feature_label_col or _find_column(
        features,
        ["label", "true_label", "y_true"],
        "feature label",
    )
    prediction_label_col = prediction_label_col or _find_column(
        predictions,
        ["label", "true_label", "y_true"],
        "prediction true label",
    )
    prediction_col = prediction_col or _find_column(
        predictions,
        ["predictions", "prediction", "pred_label", "y_pred"],
        "prediction",
    )

    missing_join_columns = [
        column
        for column in join_on
        if column not in features.columns or column not in predictions.columns
    ]
    if missing_join_columns:
        raise ValueError(f"Join columns missing in one of the files: {missing_join_columns}")

    prediction_columns = list(dict.fromkeys(join_on + [prediction_label_col, prediction_col]))
    predictions = predictions[prediction_columns].copy()

    merged = features.merge(
        predictions,
        on=join_on,
        how=join_how,
        suffixes=("", "_prediction_file"),
    )

    label_values = _normalize_text(merged[feature_label_col])
    prediction_values = _normalize_text(merged[prediction_col])
    has_prediction = merged[prediction_col].notna()

    merged["correct_prediction"] = 0
    merged.loc[has_prediction & (label_values == prediction_values), "correct_prediction"] = 1

    return merged, {
        "feature_label_col": feature_label_col,
        "prediction_label_col": prediction_label_col,
        "prediction_col": prediction_col,
    }


def add_correct_prediction(
    features_path,
    predictions_path,
    output_path=None,
    join_on=None,
    join_how="inner",
    feature_label_col=None,
    prediction_label_col=None,
    prediction_col=None,
):
    """Merge explanatory features with model predictions and add correctness.

    Args:
        features_path: Excel file containing explanatory features.
        predictions_path: CSV or Excel file containing model predictions.
        output_path: Optional output Excel path or directory.
        join_on: Columns used to merge both files. Defaults to ``["term"]``.
        join_how: Pandas merge mode, usually ``"inner"`` or ``"left"``.
        feature_label_col: True-label column in the features file.
        prediction_label_col: True-label column in the predictions file.
        prediction_col: Predicted-label column in the predictions file.

    Returns:
        pandas.DataFrame: Merged table with ``correct_prediction``.
    """
    join_on = join_on or ["term"]

    _log(f"Reading features: {features_path}")
    features = _read_table(features_path)
    _log(f"Reading predictions: {predictions_path}")
    predictions = _read_table(predictions_path)

    merged, columns = _add_correct_prediction_to_frames(
        features=features,
        predictions=predictions,
        join_on=join_on,
        join_how=join_how,
        feature_label_col=feature_label_col,
        prediction_label_col=prediction_label_col,
        prediction_col=prediction_col,
    )

    _log(f"Join columns: {join_on}")
    _log(f"Join mode: {join_how}")
    _log(f"Feature label column: {columns['feature_label_col']}")
    _log(f"Prediction label column: {columns['prediction_label_col']}")
    _log(f"Prediction column: {columns['prediction_col']}")

    output_path = _resolve_output_path(features_path, predictions_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Saving output: {output_path}")
    merged.to_excel(output_path, index=False)

    _log(f"Rows features: {len(features)}")
    _log(f"Rows predictions: {len(predictions)}")
    _log(f"Rows output: {len(merged)}")
    _log(f"Correct predictions: {int(merged['correct_prediction'].sum())}")
    _log("Done")

    return merged


def _extract_seed(seed_dir):
    match = re.search(r"seed[_-]?(\d+)", seed_dir.name)
    if not match:
        raise ValueError(f"Could not extract seed from directory name: {seed_dir}")
    return int(match.group(1))


def _find_prediction_file(seed_dir, pattern):
    matches = sorted(seed_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one prediction file in {seed_dir} matching "
            f"'{pattern}', found {len(matches)}"
        )
    return matches[0]


def _infer_graph_name(features_path):
    stem = Path(features_path).stem
    return stem.removesuffix("_exp_features")


def _infer_model_name(first_prediction_path):
    stem = Path(first_prediction_path).stem
    stem = re.sub(r"_seed\d+.*$", "", stem)
    return stem


def _resolve_all_seeds_output_path(output_path, graph_name, model_name):
    filename = f"{graph_name}__{model_name}__correct_prediction_all_seeds.xlsx"
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix:
            return output_path
        return output_path / filename
    return DEFAULT_OUTPUT_DIR / filename


def add_correct_prediction_all_seeds(
    features_path,
    artifacts_dir,
    output_path=None,
    graph_name=None,
    model_name=None,
    prediction_pattern="*_best_test_predictions.csv",
    join_on=None,
    join_how="inner",
    feature_label_col=None,
    prediction_label_col=None,
    prediction_col=None,
):
    """Add correctness labels for every seed prediction file in an artifacts dir."""
    join_on = join_on or ["term"]
    artifacts_dir = Path(artifacts_dir)

    _log(f"Reading features once: {features_path}")
    features = _read_table(features_path)

    seed_dirs = sorted(path for path in artifacts_dir.iterdir() if path.is_dir() and path.name.startswith("seed"))
    if not seed_dirs:
        raise ValueError(f"No seed directories found in: {artifacts_dir}")

    _log(f"Seed directories found: {len(seed_dirs)}")
    all_seed_frames = []
    first_prediction_path = None

    for seed_dir in seed_dirs:
        seed = _extract_seed(seed_dir)
        prediction_path = _find_prediction_file(seed_dir, prediction_pattern)
        if first_prediction_path is None:
            first_prediction_path = prediction_path

        _log(f"Processing seed {seed}: {prediction_path}")
        predictions = _read_table(prediction_path)
        merged, columns = _add_correct_prediction_to_frames(
            features=features,
            predictions=predictions,
            join_on=join_on,
            join_how=join_how,
            feature_label_col=feature_label_col,
            prediction_label_col=prediction_label_col,
            prediction_col=prediction_col,
        )
        merged["seed"] = seed
        all_seed_frames.append(merged)
        _log(
            f"Seed {seed}: rows={len(merged)}, "
            f"correct={int(merged['correct_prediction'].sum())}"
        )

    graph_name = graph_name or _infer_graph_name(features_path)
    model_name = model_name or _infer_model_name(first_prediction_path)
    output_path = _resolve_all_seeds_output_path(output_path, graph_name, model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_seeds = pd.concat(all_seed_frames, ignore_index=True)
    _log(f"Join columns: {join_on}")
    _log(f"Join mode: {join_how}")
    _log(f"Feature label column: {columns['feature_label_col']}")
    _log(f"Prediction label column: {columns['prediction_label_col']}")
    _log(f"Prediction column: {columns['prediction_col']}")
    _log(f"Saving all-seeds output: {output_path}")
    all_seeds.to_excel(output_path, index=False)
    _log(f"Rows output: {len(all_seeds)}")
    _log(f"Correct predictions: {int(all_seeds['correct_prediction'].sum())}")
    _log("Done")

    return all_seeds


def _parse_join_on(text):
    if text is None or text.strip() == "":
        return ["term"]
    return [column.strip() for column in text.split(",") if column.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Add correct_prediction to an explanatory feature Excel file."
    )
    parser.add_argument("--features-path", required=True, help="Explanatory features Excel path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions-path", help="Model predictions CSV/Excel path.")
    source.add_argument(
        "--artifacts-dir",
        help="Directory containing seed_* folders with prediction files.",
    )
    parser.add_argument("--output-path", default=None, help="Output Excel path or directory.")
    parser.add_argument("--graph-name", default=None, help="Graph name used in all-seeds output file.")
    parser.add_argument("--model-name", default=None, help="Model name used in all-seeds output file.")
    parser.add_argument(
        "--prediction-pattern",
        default="*_best_test_predictions.csv",
        help="Glob pattern used inside each seed directory.",
    )
    parser.add_argument("--join-on", default="term", help="Comma-separated join columns.")
    parser.add_argument(
        "--join-how",
        default="inner",
        choices=["inner", "left", "right", "outer"],
        help="Merge mode. Use inner to keep only predicted nodes.",
    )
    parser.add_argument("--feature-label-col", default=None)
    parser.add_argument("--prediction-label-col", default=None)
    parser.add_argument("--prediction-col", default=None)
    args = parser.parse_args()

    if args.artifacts_dir:
        add_correct_prediction_all_seeds(
            features_path=args.features_path,
            artifacts_dir=args.artifacts_dir,
            output_path=args.output_path,
            graph_name=args.graph_name,
            model_name=args.model_name,
            prediction_pattern=args.prediction_pattern,
            join_on=_parse_join_on(args.join_on),
            join_how=args.join_how,
            feature_label_col=args.feature_label_col,
            prediction_label_col=args.prediction_label_col,
            prediction_col=args.prediction_col,
        )
    else:
        add_correct_prediction(
            features_path=args.features_path,
            predictions_path=args.predictions_path,
            output_path=args.output_path,
            join_on=_parse_join_on(args.join_on),
            join_how=args.join_how,
            feature_label_col=args.feature_label_col,
            prediction_label_col=args.prediction_label_col,
            prediction_col=args.prediction_col,
        )


if __name__ == "__main__":
    main()
