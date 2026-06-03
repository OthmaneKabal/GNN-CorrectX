from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explanatory_features_extraction import ExplanatoryFeaturesExtraction


def _parse_csv_list(text):
    if text is None or text.strip() == "":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test global explanatory feature extraction."
    )
    parser.add_argument(
        "--graph-path",
        default=str(ROOT / "data" / "biomedical" / "text_derived" / "biomed_tdg.json"),
        help="Path to the graph JSON file.",
    )
    parser.add_argument(
        "--gs-path",
        default=str(ROOT / "data" / "biomedical" / "biomed_gs.xlsx"),
        help="Path to the gold-standard Excel file.",
    )
    parser.add_argument(
        "--features",
        default="structural,relational,distance,semantic",
        help="Comma-separated feature categories.",
    )
    parser.add_argument(
        "--relations",
        default="is",
        help="Comma-separated relation names to count for relational features.",
    )
    parser.add_argument(
        "--path-cutoff",
        type=int,
        default=4,
        help="Maximum number of edges for counting simple distance paths.",
    )
    parser.add_argument(
        "--plm",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model used by get_gdp to embed entities and predicates.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output directory or explicit .xlsx path.",
    )
    args = parser.parse_args()

    extractor = ExplanatoryFeaturesExtraction(
        graph_path=args.graph_path,
        gs_path=args.gs_path,
        features=_parse_csv_list(args.features),
        relations_to_count=_parse_csv_list(args.relations),
        path_cutoff=args.path_cutoff,
        plm=args.plm,
        output_path=args.output_path,
    )

    result = extractor.extract(save=True)
    features = result["features"]

    print("\nExtracted explanatory features:")
    print(features.head(10).to_string(index=False))
    print(f"\nrows: {len(features)}")
    print(f"columns: {features.columns.tolist()}")
    print(f"Excel: {result['excel_path']}")
    print(f"JSON: {result['json_path']}")
    print(f"Categories: {result['feature_categories']}")


if __name__ == "__main__":
    main()
