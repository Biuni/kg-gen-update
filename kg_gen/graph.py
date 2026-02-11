import json
from pydantic import BaseModel, Field
from typing import Any, Tuple, Optional


# =========================
# GRAPH DATA STRUCTURE
# =========================

class Graph(BaseModel):
    """
    Lightweight graph representation used throughout the pipeline.

    The graph is defined by:
    - Entities (nodes) with optional reasoning metadata
    - Edges (predicates)
    - Relations represented as (subject, predicate, object) triples

    Optional cluster and metadata fields can be used to store
    deduplication results or additional annotations.
    """

    # Entity name -> reasoning snippet
    entities: dict[str, str] = Field(
        ..., description="Entity name mapped to a reasoning snippet"
    )

    # Set of all edge labels
    edges: set[str] = Field(
        ..., description="All edge labels used in the graph"
    )

    # Set of (subject, predicate, object) triples
    relations: set[Tuple[str, str, str]] = Field(
        ..., description="Set of (subject, predicate, object) relations"
    )

    # Optional clusters produced by deduplication
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None

    # Optional per-entity metadata
    entity_metadata: dict[str, set[str]] | None = None

    @staticmethod
    def from_file(file_path: str) -> "Graph":
        """
        Load a graph from a JSON file.

        The method also ensures graph consistency by:
        - Adding missing entities referenced in relations
        - Adding missing edges referenced in relations

        Args:
            file_path: Path to the JSON file.

        Returns:
            A validated and normalized Graph instance.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            graph = Graph.model_validate(data)

        # Ensure all entities and edges referenced in relations exist
        for subject, predicate, obj in graph.relations:
            if subject not in graph.entities:
                graph.entities[subject] = ""

            if predicate not in graph.edges:
                graph.edges.add(predicate)

            if obj not in graph.entities:
                graph.entities[obj] = ""

        return graph

    def to_file(self, file_path: str):
        """
        Serialize the graph to a JSON file.

        Args:
            file_path: Destination path.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def stats(self, name: Optional[str] = None):
        """
        Print a human-readable summary of graph statistics.

        Args:
            name: Optional graph name used in the output.
        """
        print(
            f"{name or 'Graph'} with:\n"
            f"\t{len(self.entities)} entities\n"
            f"\t{len(self.edges)} edges\n"
            f"\t{len(self.relations)} relations"
        )
