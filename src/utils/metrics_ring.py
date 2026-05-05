"""
VIDYUT Theft Ring Metrics
===========================================================================
Graph-level metrics for community / ring detection evaluation:
modularity, ring precision/recall, size distributions.
===========================================================================
"""

from __future__ import annotations

from typing import Dict, List, Set

import networkx as nx
import numpy as np


def ring_detection_precision_recall(
    predicted_rings: List[Set[str]],
    ground_truth_rings: List[Set[str]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Precision and recall for ring detection using IoU matching.

    A predicted ring is a True Positive if its Jaccard similarity with any
    ground truth ring exceeds iou_threshold.

    Parameters
    ----------
    predicted_rings : list of sets of consumer IDs
    ground_truth_rings : list of sets of consumer IDs
    iou_threshold : float
        Minimum IoU for a match to count as TP.
    """
    tp = 0
    matched_gt: Set[int] = set()

    for pred in predicted_rings:
        for gt_idx, gt in enumerate(ground_truth_rings):
            if gt_idx in matched_gt:
                continue
            intersection = len(pred & gt)
            union = len(pred | gt)
            if union > 0 and intersection / union >= iou_threshold:
                tp += 1
                matched_gt.add(gt_idx)
                break

    precision = tp / (len(predicted_rings) + 1e-10)
    recall = tp / (len(ground_truth_rings) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "ring_precision": precision,
        "ring_recall": recall,
        "ring_f1": f1,
        "true_positives": float(tp),
        "predicted_rings": float(len(predicted_rings)),
        "ground_truth_rings": float(len(ground_truth_rings)),
    }


def graph_community_metrics(
    graph: nx.Graph,
    communities: List[Set],
) -> Dict[str, float]:
    """
    Compute modularity and community size statistics for a detected partition.
    """
    if len(communities) == 0 or graph.number_of_nodes() == 0:
        return {
            "modularity": 0.0,
            "num_communities": 0,
            "mean_community_size": 0.0,
            "max_community_size": 0,
            "min_community_size": 0,
        }

    try:
        modularity = nx.community.modularity(graph, communities)
    except Exception:
        modularity = 0.0

    sizes = [len(c) for c in communities]
    return {
        "modularity": float(modularity),
        "num_communities": len(communities),
        "mean_community_size": float(np.mean(sizes)),
        "max_community_size": int(np.max(sizes)),
        "min_community_size": int(np.min(sizes)),
    }


def anomaly_fraction_per_community(
    communities: List[Set[str]],
    anomaly_flags: Dict[str, bool],
) -> List[Dict]:
    """
    For each community, compute the fraction of members flagged as anomalous.

    Returns a list of dicts sorted by anomaly_fraction descending.
    """
    results = []
    for idx, community in enumerate(communities):
        if not community:
            continue
        flagged = sum(1 for cid in community if anomaly_flags.get(cid, False))
        fraction = flagged / len(community)
        results.append({
            "community_id": idx,
            "size": len(community),
            "anomalous_members": flagged,
            "anomaly_fraction": round(fraction, 4),
        })
    results.sort(key=lambda x: x["anomaly_fraction"], reverse=True)
    return results
