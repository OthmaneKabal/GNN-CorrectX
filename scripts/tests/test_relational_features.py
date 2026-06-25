from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explanatory_features_extraction.relational_features import (
    extract_relational_features,
)
from src.utils import get_gdp


def _parse_relations(relations_text):
    if relations_text is None or relations_text.strip() == "":
        return []
    return [relation.strip() for relation in relations_text.split(",") if relation.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test relational feature extraction on a graph and gold standard."
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
        "--relations",
        default="is",
        help="Comma-separated relation names to count, for example: is,isa,part_of.",
    )
    parser.add_argument(
        "--plm",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model used by get_gdp to embed entities and predicates.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path where extracted features will be saved.",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_path)
    gs_path = Path(args.gs_path)
    relations_to_count = _parse_relations(args.relations)

    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if not gs_path.exists():
        raise FileNotFoundError(gs_path)

    print(f"Graph: {graph_path}")
    print(f"Gold standard: {gs_path}")
    print(f"Embedding model: {args.plm}")
    print(f"Relations to count: {relations_to_count}")

    gdp, data = get_gdp(
        str(graph_path),
        plm=args.plm,
        use_types=True,
    )

    features = extract_relational_features(
        gdp,
        data,
        str(gs_path),
        relations_to_count=relations_to_count,
    )

    print("\nExtracted relational features:")
    print(features.head(10).to_string(index=False))
    print(f"\nrows: {len(features)}")
    print(f"found in graph: {features['found_in_graph'].sum()}")
    print(f"missing from graph: {(~features['found_in_graph']).sum()}")
    print(f"columns: {features.columns.tolist()}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        print(f"\nSaved CSV: {output_path}")


if __name__ == "__main__":
    main()
