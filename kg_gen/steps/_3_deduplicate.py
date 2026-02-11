import enum
from sentence_transformers import SentenceTransformer

from ..graph import Graph
from ..utils.deduplicate import run_semhash_deduplication
from ..utils.llm_deduplicate import LLMDeduplicate


class DeduplicateMethod(enum.Enum):
    """
    Enum representing the supported graph deduplication strategies.
    """

    # Deduplicate deterministically using semantic hashing.
    SEMHASH = "semhash"

    # Deduplicate by clustering nodes with embeddings and using an LLM to
    # identify duplicates within each cluster.
    LM_BASED = "lm_based"

    # Apply both semantic hashing and LM-based clustering sequentially.
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
    Deduplicate a graph using the specified strategy.

    This function supports three strategies:
    1. SEMHASH: Uses deterministic rules and semantic hashing.
    2. LM_BASED: Uses embedding-based clustering plus LLM-assisted deduplication.
    3. FULL: Runs SEMHASH first, then LM-based clustering on the reduced graph.

    Args:
        graph (Graph): Input graph to deduplicate.
        method (DeduplicateMethod): Deduplication strategy to run.
        retrieval_model (SentenceTransformer | None): Model for semantic similarity.
            Required for LM_BASED or FULL methods.
        semhash_similarity_threshold (float): Threshold for semantic hashing similarity.
            Higher values are more conservative.
        model (str | None): LLM model identifier for LM-based deduplication.
        api_key (str | None): Optional API key for the LLM provider.
        api_base (str | None): Optional API base URL for the LLM provider.
        temperature (float | None): Optional sampling temperature for the LLM.
        reasoning_effort (str | None): Optional reasoning effort for LLM deduplication.
        context (str | None): Optional context to guide LM-based deduplication.
        usage_history (list[dict] | None): Optional list to accumulate LiteLLM usage stats.

    Returns:
        Graph: Deduplicated graph. Clusters are updated if LM-based methods were applied.

    Raises:
        ValueError: If LM-based method is selected but no retrieval model is provided.
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
