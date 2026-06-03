"""Relational feature extraction for gold-standard nodes."""

from collections import Counter
import math
import re

import pandas as pd


def _log(message):
    print(f"[relational_features] {message}")


def _read_gold_standard(gs_path):
    _log(f"Reading gold standard: {gs_path}")
    gs = pd.read_excel(gs_path)
    if "term" not in gs.columns:
        raise ValueError("Gold standard file must contain a 'term' column.")
    _log(f"Gold standard rows: {len(gs)}")
    return gs


def _safe_feature_name(relation):
    name = str(relation).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "relation"


def _relation_id_to_name(gdp):
    return {relation_id: relation for relation, relation_id in gdp.predicate_to_id.items()}


def _edge_relation_name(edge_data, relation_id_to_name):
    relation_type = edge_data.get("type")
    return relation_id_to_name.get(relation_type, relation_type)


def _directed_edges(graph, node, direction):
    if graph.is_multigraph():
        if graph.is_directed() and direction == "in":
            return graph.in_edges(node, keys=True, data=True)
        if graph.is_directed() and direction == "out":
            return graph.out_edges(node, keys=True, data=True)
        return graph.edges(node, keys=True, data=True)

    if graph.is_directed() and direction == "in":
        return graph.in_edges(node, data=True)
    if graph.is_directed() and direction == "out":
        return graph.out_edges(node, data=True)
    return graph.edges(node, data=True)


def _relation_counter(graph, node, direction, relation_id_to_name):
    counter = Counter()
    for edge in _directed_edges(graph, node, direction):
        edge_data = edge[-1]
        relation_name = _edge_relation_name(edge_data, relation_id_to_name)
        counter[relation_name] += 1
    return counter


def _entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def extract_relational_features(gdp, data, gs_path, relations_to_count=None):
    """Extract relation-type features for gold-standard nodes.

    Args:
        gdp: Prepared GraphDataPreparation object with ``nxGraph`` built from
            ``prepare_graph_with_type``.
        data: Prepared PyTorch Geometric data object. Kept in the signature for
            consistency with the other feature extractors.
        gs_path: Path to the gold-standard Excel file containing a ``term`` column.
        relations_to_count: Optional list of relation names to count separately.
            Example: ``["is", "isa", "part_of"]``.

    Returns:
        pandas.DataFrame: One row per GS term with relational features.
    """
    if gdp.nxGraph is None or gdp.nodes_index is None:
        raise ValueError("gdp must be prepared before extracting relational features.")
    if not gdp.predicate_to_id:
        raise ValueError(
            "gdp.predicate_to_id is empty. Prepare the graph with relation types first."
        )

    graph = gdp.nxGraph
    gs = _read_gold_standard(gs_path).copy()
    relations_to_count = relations_to_count or []
    relation_id_to_name = _relation_id_to_name(gdp)

    _log(f"Graph nodes: {graph.number_of_nodes()}")
    _log(f"Graph edges: {graph.number_of_edges()}")
    _log(f"Known relation types: {len(relation_id_to_name)}")
    _log(f"Relations counted separately: {relations_to_count}")

    rows = []
    for _, gs_row in gs.iterrows():
        term = gs_row["term"]
        graph_idx = gdp.nodes_index.get(term)
        found = term in graph

        if found:
            in_counter = _relation_counter(graph, term, "in", relation_id_to_name)
            out_counter = _relation_counter(graph, term, "out", relation_id_to_name)
            total_counter = in_counter + out_counter
        else:
            in_counter = Counter()
            out_counter = Counter()
            total_counter = Counter()

        row = gs_row.to_dict()
        row.update(
            {
                "graph_idx": graph_idx,
                "found_in_graph": found,
                "nb_diverse_relations_in": len(in_counter),
                "nb_diverse_relations_out": len(out_counter),
                "nb_diverse_relations_total": len(total_counter),
                "relational_entropy": _entropy(total_counter),
            }
        )

        for relation in relations_to_count:
            feature_name = _safe_feature_name(relation)
            row[f"nb_{feature_name}_in"] = in_counter.get(relation, 0)
            row[f"nb_{feature_name}_out"] = out_counter.get(relation, 0)
            row[f"nb_{feature_name}_total"] = total_counter.get(relation, 0)

        rows.append(row)

    features = pd.DataFrame(rows)
    _log("Done")
    return features
