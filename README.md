# GNN-CorrectX

This repository contains the code and resources associated with **GNN-CorrectX**, the meta-explainability framework used to analyse when GNN predictions are correct or incorrect.

![GNN-CorrectX overview](gnn-correctX.pdf)

## 1. Requirements

Create a Python environment and install the required packages:

```bash
pip install -r requirements_server_ml.txt
```
> **Note.** Download or copy the data archive from: "", then extract it directly into this directory:
>
> ```text
> data/
>   biomedical/
>   dbpedia_174/
> ```

## 2. Data

Download or copy the data archive from: "ici link", then extract it directly into `data/`.

After extraction, the expected structure is:

```text
data/
  biomedical/
  dbpedia_174/
```

The archive should contain the KGs, gold standards, and GNN prediction artifacts required by the configuration files under `config/`.

## 3. Explanatory Features and Preprocessed ML Inputs

We provide the precomputed explanatory features and the preprocessed ML tables used in the paper. They are expected under:

```text
outputs/explanatory_features/
outputs/ml_explainer/
outputs/ml_explainer/preprocessed/
```

If you want to recompute them from the KGs, GS files, and GNN predictions, run:

```bash
python scripts/run_explanatory_features.py
python scripts/run_ml_explainer_correct_predictions.py
python scripts/run_ml_preprocess.py
```

The outputs will be written to:

```text
outputs/explanatory_features/
outputs/ml_explainer/
outputs/ml_explainer/preprocessed/
```
## 4. Machine Learning Meta-Models

To reproduce the paper results for the selected meta-models using the best hyperparameters found in our search, run:

```bash
python scripts/evaluate_selected_random_forest.py
python scripts/evaluate_selected_gradient_boosting.py --models xgboost,lightgbm --model-jobs 16
```

The main outputs are written to:

```text
outputs/reproducibility/selected_random_forest_metrics.xlsx
outputs/reproducibility/selected_gradient_boosting_metrics.xlsx
```

These files contain the OOF metrics, including accuracy, F1-macro, F1-weighted, class-specific F1 scores, and the corresponding OOF predictions.

## 5. Hyperparameter Search

If you want to rerun the global hyperparameter selection for the three meta-models, run:

```bash
python scripts/search_meta_model_hyperparameters.py --models random_forest,xgboost,lightgbm --n-folds 5 --n-configs 15 --ml-seeds 42,123,456,789,2024 --model-jobs 16
```

The outputs will be written to:

```text
outputs/reproducibility/meta_model_hyperparameter_search/
```

The selected global model and hyperparameters are saved in:

```text
outputs/reproducibility/meta_model_hyperparameter_search/selected_global_ml_model_and_hyperparams.json
```

## 6. SHAP Importance and Beeswarm Analyses

To compute SHAP values for instances correctly predicted by the meta-model, run:

```bash
python scripts/compute_shap_meta_correct_instances.py --model-jobs 8
```

The outputs will be written to:

```text
outputs/reproducibility/shap_meta_correct_instances/
```







