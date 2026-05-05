"""
VIDYUT Part B — Theft Ring Detector
===========================================================================
Stage 3: NetworkX + Louvain community detection.
Nodes = consumers. Edges created when consumers share:
  (a) the same transformer_id, OR
  (b) the same geohash prefix (≈500 m radius).

Communities where ≥ 60% of members are anomalous → RING ALERT.
===========================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import geohash2
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.metrics_ring import anomaly_fraction_per_community, graph_community_metrics

log = get_logger("vidyut.ring_detector")
settings = get_settings()


class TheftRingDetector:
    """
    Detects organised theft rings via community detection on a consumer graph.

    Graph construction rules:
    1. All consumers on the same transformer share edges (weight=1.0).
    2. Consumers within the same geohash cell (precision 6 ≈ 610×1222 m)
       share edges (weight=0.5).
    3. Louvain community detection partitions the graph.
    4. Communities with anomaly_fraction ≥ threshold → Ring Alert.
    """

    def __init__(
        self,
        geohash_precision: int = 6,
        anomaly_threshold: float = 0.60,
        min_community_size: int = 3,
        louvain_resolution: float = 1.0,
        louvain_seed: int = 42,
    ) -> None:
        self.geohash_precision = geohash_precision
        self.anomaly_threshold = anomaly_threshold
        self.min_community_size = min_community_size
        self.louvain_resolution = louvain_resolution
        self.louvain_seed = louvain_seed

        self.graph: Optional[nx.Graph] = None
        self.communities_: Optional[List[Set]] = None
        self.ring_alerts_: Optional[List[Dict]] = None

    def build_graph(
        self,
        consumers_df: pd.DataFrame,
        consumer_col: str = "consumer_id",
        lat_col: str = "lat",
        lon_col: str = "lon",
        transformer_col: str = "transformer_id",
    ) -> nx.Graph:
        """
        Build the consumer network graph.

        Parameters
        ----------
        consumers_df : pd.DataFrame
            Must contain consumer_col. Optionally lat, lon, transformer_id.
        """
        G = nx.Graph()

        # Add nodes
        for _, row in consumers_df.iterrows():
            attrs = {"consumer_id": str(row[consumer_col])}
            if lat_col in row and lon_col in row and pd.notna(row[lat_col]):
                attrs["lat"] = float(row[lat_col])
                attrs["lon"] = float(row[lon_col])
                attrs["geohash"] = geohash2.encode(
                    float(row[lat_col]),
                    float(row[lon_col]),
                    precision=self.geohash_precision,
                )
            if transformer_col in row and pd.notna(row.get(transformer_col)):
                attrs["transformer_id"] = str(row[transformer_col])
            G.add_node(str(row[consumer_col]), **attrs)

        log.info("Graph nodes added: %d consumers", G.number_of_nodes())

        # Edges by transformer
        if transformer_col in consumers_df.columns:
            transformer_groups = consumers_df.groupby(transformer_col)[consumer_col].apply(list)
            edge_count = 0
            for transformer_id, members in transformer_groups.items():
                members = [str(m) for m in members]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if not G.has_edge(members[i], members[j]):
                            G.add_edge(members[i], members[j], weight=1.0,
                                       edge_type="transformer")
                            edge_count += 1
            log.info("Transformer edges added: %d", edge_count)

        # Edges by geohash
        if lat_col in consumers_df.columns and lon_col in consumers_df.columns:
            geohash_groups: Dict[str, List[str]] = {}
            for node, data in G.nodes(data=True):
                gh = data.get("geohash")
                if gh:
                    geohash_groups.setdefault(gh, []).append(node)

            edge_count = 0
            for gh, members in geohash_groups.items():
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if not G.has_edge(members[i], members[j]):
                            G.add_edge(members[i], members[j], weight=0.5,
                                       edge_type="geohash")
                            edge_count += 1
            log.info("Geohash edges added: %d", edge_count)

        log.info(
            "Graph built: %d nodes, %d edges, density=%.4f",
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.density(G) if G.number_of_nodes() > 1 else 0,
        )
        self.graph = G
        return G

    def detect_communities(self) -> List[Set]:
        """
        Run Louvain community detection on the graph.

        Returns list of sets, each set containing consumer_id strings.
        """
        if self.graph is None:
            raise RuntimeError("Call build_graph() first.")

        log.info(
            "Running Louvain community detection (resolution=%.2f)…",
            self.louvain_resolution,
        )
        communities = louvain_communities(
            self.graph,
            weight="weight",
            resolution=self.louvain_resolution,
            seed=self.louvain_seed,
        )
        # Filter tiny communities
        communities = [c for c in communities if len(c) >= self.min_community_size]
        self.communities_ = communities

        metrics = graph_community_metrics(self.graph, communities)
        log.info(
            "Communities found: %d | modularity=%.4f | "
            "mean_size=%.1f | max_size=%d",
            metrics["num_communities"],
            metrics["modularity"],
            metrics["mean_community_size"],
            metrics["max_community_size"],
        )
        return communities

    def identify_rings(
        self,
        anomaly_flags: Dict[str, bool],
    ) -> List[Dict]:
        """
        Identify theft rings from detected communities.

        Parameters
        ----------
        anomaly_flags : dict
            Maps consumer_id → bool (True = flagged as anomalous by Stage 2).

        Returns
        -------
        List of ring alert dicts, sorted by anomaly_fraction descending.
        """
        if self.communities_ is None:
            raise RuntimeError("Call detect_communities() first.")

        community_stats = anomaly_fraction_per_community(
            self.communities_, anomaly_flags
        )

        ring_alerts = []
        for stat in community_stats:
            if stat["anomaly_fraction"] >= self.anomaly_threshold:
                community = list(self.communities_[stat["community_id"]])
                ring_alerts.append({
                    "ring_id": f"RING_{stat['community_id']:04d}",
                    "community_id": stat["community_id"],
                    "size": stat["size"],
                    "anomalous_members": stat["anomalous_members"],
                    "anomaly_fraction": stat["anomaly_fraction"],
                    "members": community,
                    "severity": "HIGH" if stat["anomaly_fraction"] >= 0.80 else "MEDIUM",
                })

        self.ring_alerts_ = ring_alerts
        log.info(
            "Theft rings identified: %d (out of %d communities)",
            len(ring_alerts), len(self.communities_),
        )
        return ring_alerts

    def run_full_detection(
        self,
        consumers_df: pd.DataFrame,
        anomaly_flags: Dict[str, bool],
        consumer_col: str = "consumer_id",
        lat_col: str = "lat",
        lon_col: str = "lon",
        transformer_col: str = "transformer_id",
    ) -> List[Dict]:
        """
        End-to-end pipeline: build graph → detect communities → identify rings.
        """
        self.build_graph(consumers_df, consumer_col, lat_col, lon_col, transformer_col)
        self.detect_communities()
        return self.identify_rings(anomaly_flags)

    def get_graph_data_for_visualisation(self) -> Dict:
        """
        Export graph data for Streamlit network visualisation.

        Returns
        -------
        dict with "nodes" and "edges" lists.
        """
        if self.graph is None:
            return {"nodes": [], "edges": []}

        ring_member_ids: Set[str] = set()
        if self.ring_alerts_:
            for ring in self.ring_alerts_:
                ring_member_ids.update(ring["members"])

        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": node_id,
                "lat": data.get("lat", 12.97),
                "lon": data.get("lon", 77.59),
                "geohash": data.get("geohash", ""),
                "transformer_id": data.get("transformer_id", ""),
                "is_ring_member": node_id in ring_member_ids,
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
                "edge_type": data.get("edge_type", "unknown"),
            })

        return {"nodes": nodes, "edges": edges}
