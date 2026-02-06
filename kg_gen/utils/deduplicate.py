import unicodedata
from ..graph import Graph
from semhash import SemHash
import inflect


class DeduplicateList:
    inflect_engine: inflect.engine
    original_map: dict[str, str]
    items_map: dict[str, str]
    duplicates: dict[str, str]
    deduplicated: list[str]

    # Stats values
    total_items: int
    deduplicated_items: int
    duplicate_items: int
    reduction: float

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.inflect_engine = inflect.engine()
        self.original_map = {}
        self.items_map = {}
        self.duplicates = {}
        self.deduplicated = []

    def normalize(self, text: str) -> str:
        """
        Normalize a text.
        """
        return unicodedata.normalize("NFKC", text)

    def singularize(self, text: str) -> str:
        """
        Singularize a text.
        """
        # singularize each token when it looks like a plural noun
        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def deduplicate(self, items: list[str]) -> list[str]:
        """
        Deduplicate a list of items using semantic hashing.
        Before deduplication, items are normalized and singularized.

        Args:
            items: List of items to deduplicate

        Returns:
            List of deduplicated items
        """
        self.total_items = len(items)

        # Normalize and singularize each string
        normalized_items = set()
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            self.original_map[item] = singular
            self.items_map[singular] = item
            normalized_items.add(singular)

        # Deduplicate the normalized strings
        semhash = SemHash.from_records(records=list(normalized_items))
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        selected = getattr(deduplication_result, "selected", None)
        if selected is None:
            selected = getattr(deduplication_result, "deduplicated", None)
        if selected is None:
            # Fallback to no-op if API shape is unknown.
            selected = list(normalized_items)

        duplicates_raw = getattr(deduplication_result, "duplicates", None)
        if duplicates_raw is None:
            duplicates_raw = []

        self.deduplicated_items = len(selected)
        if duplicates_raw:
            self.duplicate_items = len(duplicates_raw)
        else:
            self.duplicate_items = max(self.total_items - self.deduplicated_items, 0)
        self.reduction = (
            (self.duplicate_items / self.total_items) * 100
            if self.total_items > 0
            else 0.0
        )

        # Map back to original strings
        for duplicate in duplicates_raw:
            original = None
            duplicate_value = None
            if hasattr(duplicate, "record") and hasattr(duplicate, "duplicates"):
                original = duplicate.record
                if (
                    duplicate.duplicates
                    and len(duplicate.duplicates) > 0
                    and len(duplicate.duplicates[0]) > 0
                ):
                    duplicate_value = duplicate.duplicates[0][0]
            elif isinstance(duplicate, dict):
                original = duplicate.get("record")
                dups = duplicate.get("duplicates")
                if isinstance(dups, (list, tuple)) and dups:
                    first = dups[0]
                    if isinstance(first, (list, tuple)) and first:
                        duplicate_value = first[0]
            elif isinstance(duplicate, (list, tuple)) and len(duplicate) >= 2:
                original = duplicate[0]
                dups = duplicate[1]
                if isinstance(dups, (list, tuple)) and dups:
                    duplicate_value = dups[0]

            if original is not None and duplicate_value is not None:
                if duplicate_value in self.items_map:
                    self.items_map[original] = self.items_map[duplicate_value]
                if original not in self.duplicates:
                    self.duplicates[original] = duplicate_value

        self.deduplicated = selected

    def stats(self) -> str:
        return f"Total items: {self.total_items}; Deduplicated items: {self.deduplicated_items}; Duplicate items: {self.duplicate_items}; Reduction: {self.reduction:.1f}"


def run_semhash_deduplication(
    graph: Graph,
    similarity_threshold: float = 0.95,
) -> Graph:
    """
    Deduplicate the graph.
    """
    # Deduplicate each graph components
    entities_dedup = DeduplicateList(similarity_threshold)
    entities_dedup.deduplicate(graph.entities)
    edges_dedup = DeduplicateList(similarity_threshold)
    edges_dedup.deduplicate(graph.edges)

    def _get_relation(relation: list[str]) -> list[str]:
        """
        Get the transformed relation.
        """
        # Handle case where entity might not be in original_map due to normalization
        first_entity_original = relation[0]
        if first_entity_original in entities_dedup.original_map:
            first_entity = entities_dedup.items_map[
                entities_dedup.original_map[first_entity_original]
            ]
        else:
            # If not found, use the original entity (it might have been normalized differently)
            first_entity = first_entity_original

        second_entity_original = relation[2]
        if second_entity_original in entities_dedup.original_map:
            second_entity = entities_dedup.items_map[
                entities_dedup.original_map[second_entity_original]
            ]
        else:
            # If not found, use the original entity
            second_entity = second_entity_original

        edge_original = relation[1]
        if edge_original in edges_dedup.original_map:
            edge = edges_dedup.items_map[edges_dedup.original_map[edge_original]]
        else:
            # If not found, use the original edge
            edge = edge_original

        return [first_entity, edge, second_entity]

    # Deduplicate the graph
    new_entities = [
        entities_dedup.items_map[item] for item in entities_dedup.deduplicated
    ]
    new_edges = [edges_dedup.items_map[item] for item in edges_dedup.deduplicated]
    new_relations = [_get_relation(relation) for relation in graph.relations]

    # Remove duplicate relations
    new_relations = list(set(tuple(relation) for relation in new_relations))

    # Update entity_metadata keys to match deduplicated entity names
    new_entity_metadata: dict[str, set[str]] | None = None
    if graph.entity_metadata:
        new_entity_metadata = {}
        for original_entity, metadata_set in graph.entity_metadata.items():
            if original_entity in entities_dedup.original_map:
                deduped_entity = entities_dedup.items_map[
                    entities_dedup.original_map[original_entity]
                ]
            else:
                deduped_entity = original_entity
            # Merge metadata sets when entities are deduplicated together
            if deduped_entity in new_entity_metadata:
                new_entity_metadata[deduped_entity].update(metadata_set)
            else:
                new_entity_metadata[deduped_entity] = metadata_set.copy()

    return Graph(
        entities=new_entities,
        edges=new_edges,
        relations=new_relations,
        entity_metadata=new_entity_metadata,
    )
