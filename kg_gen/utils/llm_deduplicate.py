from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pydantic import BaseModel
import litellm

from ..graph import Graph


# =========================
# LLM RESPONSE SCHEMA
# =========================

class DeduplicateResponse(BaseModel):
    """
    Schema for the LLM response used during deduplication.

    Attributes:
        duplicates: List of items considered duplicates of the input item.
        alias: Canonical representative name for the cluster.
    """

    duplicates: List[str]
    alias: str


# =========================
# LLM-BASED DEDUPLICATOR
# =========================

class LLMDeduplicate:
    """
    Deduplicate graph entities and edges using a hybrid approach:
    - Dense embeddings (SentenceTransformer)
    - Sparse retrieval (BM25)
    - KMeans clustering for scalability
    - LLM-based semantic judgment for final deduplication

    The result is a deduplicated graph with explicit cluster mappings.
    """

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(
        self,
        retrieval_model: SentenceTransformer,
        graph: Graph,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        context: Optional[str] = None,
        usage_history: Optional[list[dict]] = None,
    ):
        """
        Initialize the LLM deduplicator.

        Args:
            retrieval_model: SentenceTransformer used for embeddings.
            graph: Input graph to deduplicate.
            model: LLM identifier.
            api_key: API key for the LLM provider.
            api_base: Optional custom API base URL.
            temperature: Sampling temperature for the LLM.
            reasoning_effort: Optional reasoning configuration.
            context: Optional system-level context.
            usage_history: Optional usage tracking.
        """
        self.graph = graph

        # Graph primitives
        self.nodes = list(graph.entities.keys())
        self.edges = list(graph.edges)
        self.relations = set(tuple(r) for r in graph.relations)

        # Optional pre-existing clusters
        self.node_clusters = graph.entity_clusters or []
        self.edge_clusters = graph.edge_clusters or []

        # Models & configuration
        self.retrieval_model = retrieval_model
        self.model = model or "openai/gpt-4o"
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = 0.0 if temperature is None else temperature
        self.reasoning_effort = reasoning_effort
        self.context = context
        self.usage_history = usage_history

        # Pre-compute embeddings for nodes and edges
        self.node_embeddings = retrieval_model.encode(
            self.nodes, show_progress_bar=False
        )
        self.edge_embeddings = retrieval_model.encode(
            self.edges, show_progress_bar=False
        )

        # BM25 indices for lexical retrieval
        self.node_bm25 = BM25Okapi([n.lower().split() for n in self.nodes])
        self.edge_bm25 = BM25Okapi([e.lower().split() for e in self.edges])

    # =========================
    # CLUSTERING (REQUIRED)
    # =========================

    def cluster(self, cluster_size: int = 128):
        """
        Cluster nodes and edges using KMeans.

        This step is required to limit the number of LLM calls by
        deduplicating within reasonably sized semantic clusters.

        Args:
            cluster_size: Approximate number of items per cluster.
        """

        def _cluster(items, embeddings):
            """
            Internal helper to cluster a list of items using KMeans.

            Args:
                items: List of items to cluster.
                embeddings: Corresponding embedding vectors.

            Returns:
                List of clusters, each a list of items.
            """
            if len(items) == 0:
                return []

            n_clusters = max(1, len(items) // cluster_size)
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="random",
                n_init=1,
                max_iter=20,
                tol=0.0,
                algorithm="lloyd",
            )
            kmeans.fit(embeddings.astype(np.float32))

            clusters = [[] for _ in range(n_clusters)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(items[idx])

            return clusters

        if not self.node_clusters:
            self.node_clusters = _cluster(
                self.nodes, self.node_embeddings
            )

        if not self.edge_clusters:
            self.edge_clusters = _cluster(
                self.edges, self.edge_embeddings
            )

    # =========================
    # RETRIEVAL
    # =========================

    def get_relevant_items(
        self, query: str, top_k: int, type: str
    ) -> list[str]:
        """
        Retrieve relevant candidates for a given item using
        a hybrid BM25 + cosine similarity approach.

        Args:
            query: Query string.
            top_k: Number of candidates to retrieve.
            type: Either "node" or "edge".

        Returns:
            List of candidate items.
        """
        tokens = query.lower().split()

        if type == "node":
            bm25 = self.node_bm25
            embeddings = self.node_embeddings
            items = self.nodes
        else:
            bm25 = self.edge_bm25
            embeddings = self.edge_embeddings
            items = self.edges

        # Sparse lexical scores
        bm25_scores = bm25.get_scores(tokens)

        # Dense semantic scores
        query_emb = self.retrieval_model.encode(
            [query], show_progress_bar=False
        )
        emb_scores = cosine_similarity(
            query_emb, embeddings
        ).flatten()

        # Hybrid scoring
        scores = 0.5 * bm25_scores + 0.5 * emb_scores
        idx = np.argsort(scores)[::-1][:top_k]

        return [items[i] for i in idx]

    # =========================
    # CLUSTER DEDUPLICATION
    # =========================

    def deduplicate_cluster(
        self, cluster: list[str], type: str
    ) -> tuple[set[str], dict[str, set[str]]]:
        """
        Deduplicate a single cluster using the LLM.

        Args:
            cluster: Cluster of items to deduplicate.
            type: Either "node" or "edge".

        Returns:
            - Set of representative items.
            - Mapping representative -> set of clustered items.
        """
        cluster = cluster.copy()
        representatives: set[str] = set()
        clusters: dict[str, set[str]] = {}

        while cluster:
            item = cluster.pop()

            # Retrieve candidate duplicates
            relevant = self.get_relevant_items(
                item, 16, type
            )

            # Build strict JSON schema for the LLM output
            schema = DeduplicateResponse.model_json_schema()
            schema["additionalProperties"] = False

            kwargs = {
                "model": self.model,
                "input": [
                    {
                        "role": "system",
                        "content": "Find duplicates and an alias.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Item: {item}\nCandidates:\n"
                            + "\n".join(relevant)
                        ),
                    },
                ],
                "temperature": self.temperature,
                "text": {
                    "format": {
                        "name": "deduplicate_response",
                        "type": "json_schema",
                        "schema": schema,
                        "strict": True,
                    }
                },
            }

            # Optional API configuration
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.reasoning_effort:
                kwargs["reasoning"] = {
                    "effort": self.reasoning_effort
                }

            # Call the LLM
            response = litellm.responses(**kwargs)
            raw = response.output[-1].content[0].text
            parsed = DeduplicateResponse.model_validate_json(raw)

            alias = parsed.alias
            representatives.add(alias)
            clusters.setdefault(alias, {item})

            # Remove duplicates from the working cluster
            for dup in parsed.duplicates:
                if dup in cluster:
                    cluster.remove(dup)
                    clusters[alias].add(dup)

        return representatives, clusters

    # =========================
    # FINAL DEDUPLICATION
    # =========================

    def deduplicate(self) -> Graph:
        """
        Perform full graph deduplication.

        Returns:
            A new Graph instance with:
            - Deduplicated entities and edges
            - Rewritten relations
            - Explicit entity and edge clusters
        """
        entities: set[str] = set()
        edges: set[str] = set()
        entity_clusters: dict[str, set[str]] = {}
        edge_clusters: dict[str, set[str]] = {}

        # Parallelize cluster-level deduplication
        pool = ThreadPoolExecutor(max_workers=32)

        node_futures = [
            pool.submit(
                self.deduplicate_cluster, c, "node"
            )
            for c in self.node_clusters
        ]
        edge_futures = [
            pool.submit(
                self.deduplicate_cluster, c, "edge"
            )
            for c in self.edge_clusters
        ]

        for f in node_futures:
            reps, clusters = f.result()
            entities.update(reps)
            entity_clusters.update(clusters)

        for f in edge_futures:
            reps, clusters = f.result()
            edges.update(reps)
            edge_clusters.update(clusters)

        # Rewrite relations using deduplicated representatives
        new_relations: set[tuple[str, str, str]] = set()

        for s, p, o in self.relations:
            for rep, cl in entity_clusters.items():
                if s in cl:
                    s = rep
                if o in cl:
                    o = rep
            for rep, cl in edge_clusters.items():
                if p in cl:
                    p = rep
            new_relations.add((s, p, o))

        # Graph requires a dict for entities, not a set
        entity_dict = {e: "" for e in entities}  # placeholder reasoning
        edge_set = set(edges)

        return Graph(
            entities=entity_dict,
            edges=edge_set,
            relations=new_relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
            entity_metadata=None,
        )
