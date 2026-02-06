import enum
from sentence_transformers import SentenceTransformer

from ..graph import Graph
from ..utils.deduplicate import run_semhash_deduplication
from ..utils.llm_deduplicate import LLMDeduplicate


class DeduplicateMethod(enum.Enum):
    """Supported deduplication strategies."""

    # Deduplicate using deterministic rules and semantic hashing.
    SEMHASH = "semhash"
    # Deduplicate using KNN clustering + intra-cluster LM deduplication.
    LM_BASED = "lm_based"
    # Deduplicate using both semantic hashing and LM-based clustering.
    FULL = "full"


def run_deduplication(
    graph: Graph,
    method: DeduplicateMethod = DeduplicateMethod.FULL,
    retrieval_model: SentenceTransformer | None = None,
    semhash_similarity_threshold: float = 0.95,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    context: str | None = None,
    usage_history: list[dict] | None = None,
) -> Graph:
    """
    Run graph deduplication with the selected strategy.

    Args:
        - graph: Input Graph to deduplicate.
        - method: Deduplication strategy to run.
        - retrieval_model: SentenceTransformer used for semantic similarity.
            Required for LM-based and FULL methods.
        - semhash_similarity_threshold: Similarity threshold used by semantic
            hashing; higher values are more conservative.
        - model: LLM model identifier for LM-based deduplication.
        - api_key: Optional API key override for the LLM provider.
        - api_base: Optional API base URL override for the LLM provider.
        - temperature: Optional sampling temperature override.
        - reasoning_effort: Optional reasoning effort for supported providers.
        - context: Optional context string to guide LM-based deduplication.
        - usage_history: Optional list to accumulate LiteLLM usage dicts.

    Returns:
        Graph: Deduplicated graph with updated clusters when applicable.

    Raises:
        ValueError: If an LM-based method is selected without a retrieval model.
    """
    # LM-based methods require a retrieval model for similarity search.
    if method != DeduplicateMethod.SEMHASH and retrieval_model is None:
        raise ValueError("No retrieval model provided")

    if method == DeduplicateMethod.SEMHASH:
        # Only apply semantic hashing.
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
    elif method == DeduplicateMethod.LM_BASED:
        # Cluster with embeddings and deduplicate within clusters using LM.
        llm_deduplicate = LLMDeduplicate(
            retrieval_model,
            graph,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            context=context,
            usage_history=usage_history,
        )
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()
    elif method == DeduplicateMethod.FULL:
        # Run semantic hashing first, then LM-based clustering on the reduced graph.
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
        llm_deduplicate = LLMDeduplicate(
            retrieval_model,
            deduplicated_graph,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            context=context,
            usage_history=usage_history,
        )
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()

    # Return the final deduplicated graph.
    return deduplicated_graph
