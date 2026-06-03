"""Structural feature extraction for gold-standard nodes."""

import pandas as pd
import networkx as nx


def _log(message):
    print(f"[structural_features] {message}")


def _read_gold_standard(gs_path):
    _log(f"Reading gold standard: {gs_path}")
    gs = pd.read_excel(gs_path)
    if "term" not in gs.columns:
        raise ValueError("Gold standard file must contain a 'term' column.")
    _log(f"Gold standard rows: {len(gs)}")
    return gs


def _degree_maps(graph):
    if graph.is_directed():
        return dict(graph.in_degree()), dict(graph.out_degree())

    degrees = dict(graph.degree())
    return degrees, degrees


def _closeness_centrality(graph):
    try:
        import igraph as ig
    except ImportError:
        _log("igraph is not installed; falling back to NetworkX closeness")
        return nx.closeness_centrality(graph)

    try:
        _log("Converting NetworkX graph to igraph for closeness centrality")
        igraph_graph = ig.Graph.from_networkx(graph)
        mode = "in" if graph.is_directed() else "all"
        closeness_values = igraph_graph.closeness(mode=mode, normalized=True)

        if "_nx_name" in igraph_graph.vs.attributes():
            node_names = igraph_graph.vs["_nx_name"]
        elif "name" in igraph_graph.vs.attributes():
            node_names = igraph_graph.vs["name"]
        else:
            node_names = list(graph.nodes())

        return dict(zip(node_names, closeness_values))
    except Exception as error:
        _log(f"igraph closeness failed ({error}); falling back to NetworkX closeness")
        return nx.closeness_centrality(graph)


def extract_structural_features(gdp, data, gs_path):
    """Extract NetworkX structural features for gold-standard nodes.

    Args:
        gdp: Prepared GraphDataPreparation object. Its ``nxGraph`` and
            ``nodes_index`` attributes must already be built.
        data: Prepared PyTorch Geometric data object. Kept in the signature for
            consistency with the other feature extractors.
        gs_path: Path to the gold-standard Excel file containing a ``term`` column.

    Returns:
        pandas.DataFrame: One row per GS term with structural features.
    """
    if gdp.nxGraph is None or gdp.nodes_index is None:
        raise ValueError("gdp must be prepared before extracting structural features.")

    graph = gdp.nxGraph
    gs = _read_gold_standard(gs_path).copy()
    gs_nodes = [term for term in gs["term"].tolist() if term in graph]
    _log(f"Graph nodes: {graph.number_of_nodes()}")
    _log(f"Graph edges: {graph.number_of_edges()}")
    _log(f"GS nodes found in graph: {len(gs_nodes)} / {len(gs)}")

    _log("Computing degree features")
    in_degree, out_degree = _degree_maps(graph)
    total_degree = dict(graph.degree())

    _log("Computing PageRank")
    pagerank = nx.pagerank(graph)

    _log("Computing betweenness centrality on GS subset")
    betweenness = nx.betweenness_centrality_subset(
        graph,
        sources=gs_nodes,
        targets=gs_nodes,
    )

    _log("Computing global closeness centrality")
    closeness = _closeness_centrality(graph)

    _log("Building structural features DataFrame")
    rows = []
    for _, gs_row in gs.iterrows():
        term = gs_row["term"]
        graph_idx = gdp.nodes_index.get(term)
        found = term in graph

        row = gs_row.to_dict()
        row.update(
            {
                "graph_idx": graph_idx,
                "found_in_graph": found,
                "in_degree": in_degree.get(term),
                "out_degree": out_degree.get(term),
                "total_degree": total_degree.get(term),
                "pagerank": pagerank.get(term),
                "betweenness_centrality": betweenness.get(term),
                "closeness_centrality": closeness.get(term),
            }
        )
        rows.append(row)

    features = pd.DataFrame(rows)
    _log("Done")
    return features
