"""Semantic feature extraction for gold-standard nodes."""

import math

import pandas as pd
import torch
import torch.nn.functional as F


def _log(message):
    print(f"[semantic_features] {message}")


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


def _node_embedding(graph, node):
    if node not in graph:
        return None

    embedding = graph.nodes[node].get("emb")
    if embedding is None:
        return None

    if not isinstance(embedding, torch.Tensor):
        embedding = torch.as_tensor(embedding)

    return embedding.detach().float().reshape(-1)


def _cosine_similarity(left_embedding, right_embedding):
    if left_embedding is None or right_embedding is None:
        return math.nan

    if left_embedding.numel() == 0 or right_embedding.numel() == 0:
        return math.nan

    return F.cosine_similarity(
        left_embedding.unsqueeze(0),
        right_embedding.unsqueeze(0),
    ).item()


def _candidate_type_nodes(gs, graph):
    labels = sorted(label for label in gs["label"].dropna().unique())
    return [label for label in labels if label in graph]


def _neighbors(graph, node):
    if node not in graph:
        return []

    if graph.is_directed():
        return sorted(set(graph.predecessors(node)) | set(graph.successors(node)))

    return sorted(graph.neighbors(node))


def _nearest_type_prediction(node_embedding, type_embeddings):
    if node_embedding is None or not type_embeddings:
        return None, math.nan

    best_type = None
    best_similarity = -math.inf
    for candidate_type, candidate_embedding in type_embeddings.items():
        similarity = _cosine_similarity(node_embedding, candidate_embedding)
        if not math.isnan(similarity) and similarity > best_similarity:
            best_type = candidate_type
            best_similarity = similarity

    if best_type is None:
        return None, math.nan
    return best_type, best_similarity


def extract_semantic_features(gdp, data, gs_path):
    """Extract embedding-based semantic features for gold-standard nodes.

    The correct type node is taken from the GS ``label`` column and matched by
    name against nodes in ``gdp.nxGraph``.

    Args:
        gdp: Prepared GraphDataPreparation object. Its ``nxGraph`` must contain
            node embeddings under the ``emb`` node attribute.
        data: Prepared PyTorch Geometric data object. Kept in the signature for
            consistency with the other feature extractors.
        gs_path: Path to the gold-standard Excel file containing ``term`` and
            ``label`` columns.

    Returns:
        pandas.DataFrame: One row per GS term with semantic features.
    """
    if gdp.nxGraph is None or gdp.nodes_index is None:
        raise ValueError("gdp must be prepared before extracting semantic features.")

    graph = gdp.nxGraph
    gs = _read_gold_standard(gs_path).copy()
    candidate_types = _candidate_type_nodes(gs, graph)
    type_embeddings = {
        candidate_type: _node_embedding(graph, candidate_type)
        for candidate_type in candidate_types
    }
    type_embeddings = {
        candidate_type: embedding
        for candidate_type, embedding in type_embeddings.items()
        if embedding is not None
    }

    _log(f"Graph nodes: {graph.number_of_nodes()}")
    _log(f"Graph edges: {graph.number_of_edges()}")
    _log(f"Candidate type embeddings found: {len(type_embeddings)}")
    _log("Computing semantic features")

    rows = []
    for _, gs_row in gs.iterrows():
        term = gs_row["term"]
        correct_type = gs_row["label"]
        graph_idx = gdp.nodes_index.get(term)
        found = term in graph
        correct_type_found = correct_type in type_embeddings

        term_embedding = _node_embedding(graph, term)
        correct_type_embedding = type_embeddings.get(correct_type)

        similarity_to_type = _cosine_similarity(
            term_embedding,
            correct_type_embedding,
        )
        predicted_type, _ = _nearest_type_prediction(
            term_embedding,
            type_embeddings,
        )
        if term_embedding is None:
            plm_only_classification = None
        else:
            plm_only_classification = predicted_type == correct_type

        neighbor_similarities = [
            _cosine_similarity(term_embedding, _node_embedding(graph, neighbor))
            for neighbor in _neighbors(graph, term)
        ]
        neighbor_similarities = [
            similarity
            for similarity in neighbor_similarities
            if not math.isnan(similarity)
        ]

        if neighbor_similarities:
            neighbors_tensor = torch.tensor(neighbor_similarities, dtype=torch.float)
            avg_neighbors_similarity = neighbors_tensor.mean().item()
            std_neighbors_similarity = neighbors_tensor.std(unbiased=False).item()
        else:
            avg_neighbors_similarity = math.nan
            std_neighbors_similarity = math.nan

        other_type_similarities = [
            _cosine_similarity(term_embedding, candidate_embedding)
            for candidate_type, candidate_embedding in type_embeddings.items()
            if candidate_type != correct_type
        ]
        other_type_similarities = [
            similarity
            for similarity in other_type_similarities
            if not math.isnan(similarity)
        ]
        max_similarity_to_other_types = (
            max(other_type_similarities) if other_type_similarities else math.nan
        )

        row = gs_row.to_dict()
        row.update(
            {
                "graph_idx": graph_idx,
                "found_in_graph": found,
                "correct_type_found_in_graph": correct_type_found,
                "similarity_to_type": similarity_to_type,
                "plm_only_classification": plm_only_classification,
                "avg_neighbors_similarity": avg_neighbors_similarity,
                "std_neighbors_similarity": std_neighbors_similarity,
                "max_similarity_to_other_types": max_similarity_to_other_types,
            }
        )
        rows.append(row)

    features = pd.DataFrame(rows)
    _log("Done")
    return features
