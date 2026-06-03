"""
Utility to load a GraphDataPreparation (gdp) from a given graph JSON path.

This module uses the existing `tdg-bench/data_preprocessing/GraphDataPreparation.py`
implementation. It ensures the `tdg-bench` folder is on `sys.path` so the
internal modules can be imported as they are when running scripts from
`tdg-bench`.

Function:
 - `get_gdp(graph_path, plm=..., use_types=False, ...)` -> returns (gdp, data)

The returned `gdp` is the `GraphDataPreparation` instance; `data` is the
`torch_geometric.data.Data` object produced by `gdp.prepare_graph()` or
`gdp.prepare_graph_with_type()`.
"""
from typing import Tuple, Optional
import os
import sys


def _ensure_tdg_bench_on_path():
    # Add the tdg-bench folder to sys.path so `data_preprocessing` imports work
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # workspace root assumed two levels up from src/utils
    workspace_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    tdg_bench_path = os.path.join(workspace_root, "tdg-bench")
    if tdg_bench_path not in sys.path:
        sys.path.insert(0, tdg_bench_path)


def get_gdp(
    graph_path: str,
    plm: str = "sentence-transformers/all-MiniLM-L6-v2",
    is_directed: bool = True,
    with_self_loop: bool = False,
    emb_dim: int = 256,
    use_types: bool = False,
    entities_embd_path: Optional[str] = None,
    edges_embd_path: Optional[str] = None,
):
    """Create and prepare a GraphDataPreparation (gdp) from a graph JSON path.

    Args:
        graph_path: Path to the KG JSON file.
        plm: pre-trained language model to use for initial embeddings.
        is_directed: whether to treat the graph as directed.
        with_self_loop: keep self-loops when building the graph.
        emb_dim: embedding dimensionality used for random embeddings.
        use_types: if True, prepare graph with relation types (calls prepare_graph_with_type()).
        entities_embd_path, edges_embd_path: optional explicit embedding pickle paths.

    Returns:
        (gdp, data): the `GraphDataPreparation` instance and the prepared `Data` object.
    """
    _ensure_tdg_bench_on_path()

    # Import after sys.path setup
    from data_preprocessing.GraphDataPreparation import GraphDataPreparation

    # Derive a dataset name from the filename (used by GraphDataPreparation constructor)
    kg_name = os.path.splitext(os.path.basename(graph_path))[0]

    gdp = GraphDataPreparation(
        kg_name=kg_name,
        model_name_init=plm,
        entities_embd_path=entities_embd_path,
        edges_embd_path=edges_embd_path,
        is_directed=is_directed,
        with_self_loop=with_self_loop,
        emb_dim=emb_dim,
    )

    # Override the internal kg_path so the provided path is used directly
    gdp.kg_path = graph_path

    # Prepare the graph (with or without relation types)
    if use_types:
        data = gdp.prepare_graph_with_type()
    else:
        data = gdp.prepare_graph()

    return gdp, data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load KG and return prepared gdp/data")
    parser.add_argument("graph_path", help="Path to KG JSON file")
    parser.add_argument("--plm", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--use-types", action="store_true")
    args = parser.parse_args()

    gdp, data = get_gdp(args.graph_path, plm=args.plm, use_types=args.use_types)
    print("Prepared graph:", data)
