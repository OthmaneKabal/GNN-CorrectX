"""Distance-based feature extraction for gold-standard nodes."""

from time import perf_counter

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


def _correct_type_distance_rank(shortest_distances, correct_type, candidate_types):
    if correct_type not in shortest_distances:
        return -1

    distances = []
    for candidate_type in candidate_types:
        distance = shortest_distances.get(candidate_type)
        if distance is not None:
            distances.append((candidate_type, distance))

    if not distances:
        return -1

    distances.sort(key=lambda item: (item[1], item[0]))
    for rank, (candidate_type, _) in enumerate(distances, start=1):
        if candidate_type == correct_type:
            return rank
    return -1


def extract_distance_features(gdp, data, gs_path, path_cutoff=4, log_every=50):
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
        path_cutoff: Kept for API compatibility. Path counting was removed
            because enumerating all simple paths is too expensive on large graphs.
        log_every: Print progress every N GS rows.

    Returns:
        pandas.DataFrame: One row per GS term with distance features.
    """
    if gdp.nxGraph is None or gdp.nodes_index is None:
        raise ValueError("gdp must be prepared before extracting distance features.")

    graph = gdp.nxGraph
    _log("Converting graph to undirected view for undirected distances")
    start_time = perf_counter()
    undirected_graph = graph.to_undirected()
    _log(f"Undirected graph ready in {perf_counter() - start_time:.2f}s")

    gs = _read_gold_standard(gs_path).copy()
    candidate_types = _candidate_type_nodes(gs, graph)

    _log(f"Graph nodes: {graph.number_of_nodes()}")
    _log(f"Graph edges: {graph.number_of_edges()}")
    _log(f"Candidate type nodes found in graph: {len(candidate_types)}")
    _log("Path counting disabled: nb_paths_to_type was removed for speed")
    _log("Computing distance features")

    rows = []
    total_rows = len(gs)
    loop_start = perf_counter()
    missing_terms = 0
    missing_types = 0
    unreachable_directed = 0
    unreachable_undirected = 0

    for row_index, (_, gs_row) in enumerate(gs.iterrows(), start=1):
        row_start = perf_counter()
        term = gs_row["term"]
        correct_type = gs_row["label"]
        graph_idx = gdp.nodes_index.get(term)
        found = term in graph
        correct_type_found = correct_type in graph

        if not found:
            missing_terms += 1
        if not correct_type_found:
            missing_types += 1

        directed_distance = _shortest_path_length(graph, term, correct_type)
        if directed_distance == -1:
            unreachable_directed += 1

        undirected_distance = _shortest_path_length(
            undirected_graph,
            term,
            correct_type,
        )
        if undirected_distance == -1:
            unreachable_undirected += 1
            correct_type_distance_rank = -1
        else:
            shortest_distances = nx.single_source_shortest_path_length(
                undirected_graph,
                term,
            )
            correct_type_distance_rank = _correct_type_distance_rank(
                shortest_distances,
                correct_type,
                candidate_types,
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
            }
        )
        rows.append(row)

        if log_every and (row_index == 1 or row_index % log_every == 0 or row_index == total_rows):
            elapsed = perf_counter() - loop_start
            avg_time = elapsed / row_index
            _log(
                "Progress "
                f"{row_index}/{total_rows} "
                f"({row_index / total_rows:.1%}) | "
                f"last={perf_counter() - row_start:.2f}s | "
                f"avg={avg_time:.2f}s/row | "
                f"elapsed={elapsed:.2f}s"
            )

    features = pd.DataFrame(rows)
    _log(
        "Summary: "
        f"missing_terms={missing_terms}, "
        f"missing_types={missing_types}, "
        f"unreachable_directed={unreachable_directed}, "
        f"unreachable_undirected={unreachable_undirected}"
    )
    _log(f"Done in {perf_counter() - loop_start:.2f}s")
    return features
