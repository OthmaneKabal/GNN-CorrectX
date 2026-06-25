from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_gdp

GRAPHS = [
    ROOT / "data" / "dbpedia_174" / "curated" / "dbpedia_174_cg.json",
    ROOT / "data" / "biomedical" / "curated" / "biomed_cg.json",
    ROOT / "data" / "biomedical" / "text_derived" / "biomed_tdg.json",
]


def describe_graph(graph_path: Path, plm: str) -> None:
    print(f"\n=== {graph_path.relative_to(ROOT)} ===")
    print(f"embedding model: {plm}")

    gdp, data = get_gdp(
        str(graph_path),
        plm=plm,
        use_types=True,
    )

    print(data)
    print(f"nodes: {data.num_nodes}")
    print(f"edges: {data.edge_index.size(1)}")
    print(f"node features: {tuple(data.x.shape)}")

    if getattr(data, "edge_type", None) is not None:
        print(f"relation types: {len(gdp.predicate_to_id)}")
        print(f"edge_type shape: {tuple(data.edge_type.shape)}")

    if getattr(data, "edge_attr", None) is not None:
        print(f"edge_attr shape: {tuple(data.edge_attr.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test get_gdp on the three project graphs with Sentence-BERT embeddings."
    )
    parser.add_argument(
        "--plm",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT / Hugging Face model used to embed entities and predicates.",
    )
    args = parser.parse_args()

    for graph_path in GRAPHS:
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        describe_graph(graph_path, args.plm)


if __name__ == "__main__":
    main()
