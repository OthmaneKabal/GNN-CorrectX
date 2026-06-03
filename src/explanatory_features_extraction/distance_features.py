"""Distance-based feature extraction for gold-standard nodes."""

import networkx as nx
import pandas as pd


def _log(message):
    print(f"[distance_features] {message}")


def _read_gold_standard(gs_path):
    _log(f"Reading gold standard: {gs_path}")
    gs = pd.read_excel(gs_path)
    required_columns = {"term", "label"}
    missing_columns = required_columns - set(gs.columns)
    if missing_columns:
        raise ValueError(
            f"Gold standard file must contain columns: {sorted(missing_columns)}"
        )
    _log(f"Gold standard rows: {len(gs)}")
    return gs


def _shortest_path_length(graph, source, target):
    if source not in graph or target not in graph:
        return -1
    try:
        return nx.shortest_path_length(graph, source=source, target=target)
    except nx.NetworkXNoPath:
        return -1


def _candidate_type_nodes(gs, graph):
    labels = sorted(label for label in gs["label"].dropna().unique())
    return [label for label in labels if label in graph]


def _correct_type_distance_rank(undirected_graph, source, correct_type, candidate_types):
    if source not in undirected_graph or correct_type not in undirected_graph:
        return -1

    distances = []
    for candidate_type in candidate_types:
        distance = _shortest_path_length(undirected_graph, source, candidate_type)
        if distance != -1:
            distances.append((candidate_type, distance))

    if not distances:
        return -1

    distances.sort(key=lambda item: (item[1], item[0]))
    for rank, (candidate_type, _) in enumerate(distances, start=1):
        if candidate_type == correct_type:
            return rank
    return -1


def _count_paths_to_type(undirected_graph, source, target, cutoff):
    if source not in undirected_graph or target not in undirected_graph:
        return 0

    # NetworkX cutoff is expressed in number of edges, same unit as
    # nx.shortest_path_length.
    paths = nx.all_simple_paths(
        undirected_graph,
        source=source,
        target=target,
        cutoff=cutoff,
    )
    return sum(1 for _ in paths)


def extract_distance_features(gdp, data, gs_path, path_cutoff=4):
    """Extract graph-distance features for gold-standard nodes.

    The correct type node is taken from the GS ``label`` column and matched by
    name against nodes in ``gdp.nxGraph``.

    Args:
        gdp: Prepared GraphDataPreparation object. Its ``nxGraph`` and
            ``nodes_index`` attributes must already be built.
        data: Prepared PyTorch Geometric data object. Kept in the signature for
            consistency with the other feature extractors.
        gs_path: Path to the gold-standard Excel file containing ``term`` and
            ``label`` columns.
        path_cutoff: Maximum number of edges for counting simple paths to the correct
            type. This protects large graphs from combinatorial path explosion.

    Returns:
        pandas.DataFrame: One row per GS term with distance features.
    """
    if gdp.nxGraph is None or gdp.nodes_index is None:
        raise ValueError("gdp must be prepared before extracting distance features.")

    graph = gdp.nxGraph
    undirected_graph = graph.to_undirected()
    gs = _read_gold_standard(gs_path).copy()
    candidate_types = _candidate_type_nodes(gs, graph)

    _log(f"Graph nodes: {graph.number_of_nodes()}")
    _log(f"Graph edges: {graph.number_of_edges()}")
    _log(f"Candidate type nodes found in graph: {len(candidate_types)}")
    _log(f"Path counting cutoff: {path_cutoff}")
    _log("Computing distance features")

    rows = []
    for _, gs_row in gs.iterrows():
        term = gs_row["term"]
        correct_type = gs_row["label"]
        graph_idx = gdp.nodes_index.get(term)
        found = term in graph
        correct_type_found = correct_type in graph

        directed_distance = _shortest_path_length(graph, term, correct_type)
        undirected_distance = _shortest_path_length(
            undirected_graph,
            term,
            correct_type,
        )
        correct_type_distance_rank = _correct_type_distance_rank(
            undirected_graph,
            term,
            correct_type,
            candidate_types,
        )
        nb_paths_to_type = _count_paths_to_type(
            undirected_graph,
            term,
            correct_type,
            path_cutoff,
        )

        row = gs_row.to_dict()
        row.update(
            {
                "graph_idx": graph_idx,
                "found_in_graph": found,
                "correct_type_found_in_graph": correct_type_found,
                "directed_distance": directed_distance,
                "undirected_distance": undirected_distance,
                "correct_type_distance_rank": correct_type_distance_rank,
                "nb_paths_to_type": nb_paths_to_type,
            }
        )
        rows.append(row)

    features = pd.DataFrame(rows)
    _log("Done")
    return features
