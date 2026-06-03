"""Orchestrates explanatory feature extraction for gold-standard nodes."""

import json
from pathlib import Path

import pandas as pd

from src.utils import get_gdp


BASE_COLUMNS = ["term", "label"]
NON_FEATURE_COLUMNS = {
    "idx",
    "term",
    "cui",
    "label",
    "graph_idx",
    "found_in_graph",
    "correct_type_found_in_graph",
}


def _log(message):
    print(f"[explanatory_features] {message}")


class ExplanatoryFeaturesExtraction:
    """Extract selected explanatory feature families from one prepared graph."""

    FEATURE_EXTRACTORS = {"structural", "relational", "distance", "semantic"}

    def __init__(
        self,
        graph_path,
        gs_path,
        features,
        relations_to_count=None,
        path_cutoff=4,
        plm="sentence-transformers/all-MiniLM-L6-v2",
        use_types=True,
        output_path=None,
        **gdp_kwargs,
    ):
        self.graph_path = graph_path
        self.gs_path = gs_path
        self.features = self._normalize_features(features)
        self.relations_to_count = relations_to_count or []
        self.path_cutoff = path_cutoff
        self.plm = plm
        self.use_types = use_types
        self.output_path = output_path
        self.gdp_kwargs = gdp_kwargs
        self.gdp = None
        self.data = None
        self.feature_categories = {}

    def _normalize_features(self, features):
        if features is None:
            return list(self.FEATURE_EXTRACTORS)

        normalized = [feature.lower().strip() for feature in features]
        unknown = sorted(set(normalized) - set(self.FEATURE_EXTRACTORS))
        if unknown:
            raise ValueError(
                f"Unknown feature categories: {unknown}. "
                f"Available categories: {sorted(self.FEATURE_EXTRACTORS)}"
            )
        return normalized

    def prepare_graph(self):
        """Build the GDP object and PyG data once."""
        _log("Preparing graph once with get_gdp")
        self.gdp, self.data = get_gdp(
            self.graph_path,
            plm=self.plm,
            use_types=self.use_types,
            **self.gdp_kwargs,
        )
        return self.gdp, self.data

    def _base_dataframe(self):
        gs = pd.read_excel(self.gs_path)
        missing_columns = [column for column in BASE_COLUMNS if column not in gs.columns]
        if missing_columns:
            raise ValueError(
                f"Gold standard file must contain columns: {missing_columns}"
            )
        return gs[BASE_COLUMNS].copy()

    def _feature_columns(self, features_df):
        return [
            column
            for column in features_df.columns
            if column not in NON_FEATURE_COLUMNS
        ]

    def _extract_category(self, category):
        if category == "structural":
            from .structural_features import extract_structural_features

            return extract_structural_features(self.gdp, self.data, self.gs_path)
        if category == "relational":
            from .relational_features import extract_relational_features

            return extract_relational_features(
                self.gdp,
                self.data,
                self.gs_path,
                relations_to_count=self.relations_to_count,
            )
        if category == "distance":
            from .distance_features import extract_distance_features

            return extract_distance_features(
                self.gdp,
                self.data,
                self.gs_path,
                path_cutoff=self.path_cutoff,
            )
        if category == "semantic":
            from .semantic_features import extract_semantic_features

            return extract_semantic_features(self.gdp, self.data, self.gs_path)
        raise ValueError(f"Unsupported feature category: {category}")

    def _resolve_output_paths(self):
        graph_name = Path(self.graph_path).stem
        default_dir = Path("outputs") / "explanatory_features"

        if self.output_path is None:
            output_dir = default_dir
            excel_path = output_dir / f"{graph_name}_exp_features.xlsx"
        else:
            output_path = Path(self.output_path)
            if output_path.suffix:
                excel_path = output_path
                output_dir = excel_path.parent
            else:
                output_dir = output_path
                excel_path = output_dir / f"{graph_name}_exp_features.xlsx"

        json_path = excel_path.with_name(f"{excel_path.stem}_categories.json")
        return excel_path, json_path

    def extract(self, save=True):
        """Run selected feature extractors and optionally save Excel/JSON outputs.

        Returns:
            dict: Contains the final DataFrame, feature category mapping, and
            output file paths.
        """
        if self.gdp is None or self.data is None:
            self.prepare_graph()

        final_features = self._base_dataframe()
        self.feature_categories = {}

        for category in self.features:
            _log(f"Extracting {category} features")
            category_df = self._extract_category(category)
            feature_columns = self._feature_columns(category_df)
            self.feature_categories[category] = feature_columns
            final_features = pd.concat(
                [final_features, category_df[feature_columns]],
                axis=1,
            )

        excel_path, json_path = self._resolve_output_paths()
        if save:
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            _log(f"Saving Excel features: {excel_path}")
            final_features.to_excel(excel_path, index=False)
            _log(f"Saving feature categories JSON: {json_path}")
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(self.feature_categories, file, indent=2)

        _log("Done")
        return {
            "features": final_features,
            "feature_categories": self.feature_categories,
            "excel_path": excel_path,
            "json_path": json_path,
        }
