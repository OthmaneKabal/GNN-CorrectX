"""Prepare cleaned ML explainer datasets for model training."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _log(message):
    print(f"[prepare_ml_data] {message}")


def _validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def _stratify_or_none(y):
    if y.nunique() > 1 and y.value_counts().min() >= 2:
        return y
    return None


def _scale_as_dataframes(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_scaled, X_val_scaled, X_test_scaled


def prepare_data(
    excel_path,
    seed_gnn,
    seed_ml=42,
    train_ratio=0.60,
    val_ratio=0.10,
    test_ratio=0.30,
):
    """Prepare train/val/test splits for one GNN seed.

    Args:
        excel_path: Path to a cleaned ML Excel file.
        seed_gnn: GNN seed to select from the ``seed`` column.
        seed_ml: Random seed used for train/val/test splitting.
        train_ratio: Fraction of rows used for train.
        val_ratio: Fraction of rows used for validation.
        test_ratio: Fraction of rows used for test.

    Returns:
        tuple: ``X_train, X_val, X_test, y_train, y_val, y_test`` as pandas
        DataFrames/Series. Numeric features are standardized using statistics
        fitted on ``X_train`` only.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    _log(f"Reading dataset: {excel_path}")
    df = pd.read_excel(excel_path)
    _log(f"Rows loaded: {len(df)}")

    required_columns = {"seed", "correct_prediction"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[df["seed"] == seed_gnn].copy()
    _log(f"Rows after seed_gnn={seed_gnn} filter: {len(df)}")
    if df.empty:
        raise ValueError(f"No rows found for seed_gnn={seed_gnn}")

    y = df["correct_prediction"].astype(int)
    X = df.drop(columns=["correct_prediction", "seed"])
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        missing_counts = X.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Feature matrix contains missing values: {missing_counts}")

    _log(f"Features count: {X.shape[1]}")
    _log(f"Class distribution: {y.value_counts().to_dict()}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=train_ratio,
        random_state=seed_ml,
        stratify=_stratify_or_none(y),
    )

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=relative_val_ratio,
        random_state=seed_ml,
        stratify=_stratify_or_none(y_temp),
    )

    X_train, X_val, X_test = _scale_as_dataframes(X_train, X_val, X_test)

    _log(f"Train rows: {len(X_train)}")
    _log(f"Train class distribution: {y_train.value_counts().to_dict()}")
    _log(f"Val rows: {len(X_val)}")
    _log(f"Val class distribution: {y_val.value_counts().to_dict()}")
    _log(f"Test rows: {len(X_test)}")
    _log(f"Test class distribution: {y_test.value_counts().to_dict()}")
    _log("Done")

    return X_train, X_val, X_test, y_train, y_val, y_test
