"""Clean correct-prediction datasets before ML splitting/training."""

from pathlib import Path
import argparse

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs") / "ml_explainer" / "preprocessed"
DEFAULT_DROP_COLUMNS = [
    "term",
    "label",
    "label_prediction_file",
    "predictions",
    "closeness_centrality",
]
DEFAULT_DISTANCE_COLUMNS = [
    "directed_distance",
    "undirected_distance",
    "correct_type_distance_rank",
]


def _log(message):
    print(f"[ml_preprocess] {message}")


def _read_excel(path):
    return pd.read_excel(path)


def _resolve_output_path(input_path, output_path):
    input_path = Path(input_path)
    filename = f"{input_path.stem}_cleaned_ml.xlsx"

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix:
            return output_path
        return output_path / filename

    return DEFAULT_OUTPUT_DIR / filename


def _replacement_value(valid_values):
    if valid_values.empty:
        return 0.0

    std = valid_values.std(ddof=0)
    if pd.isna(std):
        std = 0.0
    return valid_values.max() + std


def _clean_distance_column(df, column, add_flags):
    if column not in df.columns:
        _log(f"Distance column not found, skipping: {column}")
        return df

    df[column] = pd.to_numeric(df[column], errors="coerce")
    unreachable_mask = df[column] == -1
    valid_values = df.loc[~unreachable_mask & df[column].notna(), column]
    replacement = _replacement_value(valid_values)

    if add_flags:
        df[f"{column}_unreachable"] = unreachable_mask.astype(int)

    df.loc[unreachable_mask, column] = replacement
    _log(
        f"{column}: replaced {int(unreachable_mask.sum())} unreachable values "
        f"with {replacement}"
    )
    return df


def preprocess_correct_prediction_dataset(
    input_path,
    output_path=None,
    add_flags=False,
    drop_columns=None,
    distance_columns=None,
):
    """Clean an all-seeds correct-prediction dataset for downstream ML.

    This function intentionally does not standardize features. Scaling should
    be fitted after train/test splitting to avoid leakage.
    """
    drop_columns = drop_columns or DEFAULT_DROP_COLUMNS
    distance_columns = distance_columns or DEFAULT_DISTANCE_COLUMNS

    _log(f"Reading input: {input_path}")
    df = _read_excel(input_path)
    _log(f"Input shape: {df.shape}")

    missing_required = [
        column for column in ["seed", "correct_prediction"] if column not in df.columns
    ]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    columns_to_drop = [column for column in drop_columns if column in df.columns]
    if columns_to_drop:
        _log(f"Dropping columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    for column in distance_columns:
        df = _clean_distance_column(df, column, add_flags=add_flags)

    output_path = _resolve_output_path(input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Saving output: {output_path}")
    df.to_excel(output_path, index=False)
    _log(f"Output shape: {df.shape}")
    _log("Done")
    return df


def _parse_csv_list(text):
    if text is None or text.strip() == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Clean an all-seeds correct-prediction Excel file for ML."
    )
    parser.add_argument("--input-path", required=True, help="Input all-seeds Excel path.")
    parser.add_argument("--output-path", default=None, help="Output Excel path or directory.")
    parser.add_argument(
        "--add-flags",
        action="store_true",
        help="Add *_unreachable flags for distance/rank columns before imputation.",
    )
    parser.add_argument(
        "--drop-columns",
        default=",".join(DEFAULT_DROP_COLUMNS),
        help="Comma-separated columns to drop if present.",
    )
    parser.add_argument(
        "--distance-columns",
        default=",".join(DEFAULT_DISTANCE_COLUMNS),
        help="Comma-separated columns where -1 means unreachable/missing.",
    )
    args = parser.parse_args()

    preprocess_correct_prediction_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        add_flags=args.add_flags,
        drop_columns=_parse_csv_list(args.drop_columns),
        distance_columns=_parse_csv_list(args.distance_columns),
    )


if __name__ == "__main__":
    main()
