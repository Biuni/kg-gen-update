import json
import os
from typing import Union, List, Dict, Optional
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

from .graph import Graph
from .steps._1_get_entities import get_entities
from .steps._2_get_relations import get_relations
from .steps._3_deduplicate import run_deduplication, DeduplicateMethod
from .utils.chunk_text import chunk_text
from .visualize.visualize_kg import visualize_kg

import logging
logging.disable(logging.INFO)

class KGGen:
    """
    Knowledge Graph generator and utility toolkit.

    This class orchestrates:
    - Entity extraction
    - Relation extraction
    - Optional deduplication
    - Graph aggregation and visualization
    """
    def __init__(
        self,
        model: str = "openai/gpt-4o",           
        max_tokens: int = 16000,                
        temperature: float = 0.0,               
        reasoning_effort: str = None,           
        api_key: str = None,                    
        api_base: str = None,                   
        retrieval_model: Optional[str] = None,  
        disable_cache: bool = False,            
    ):
        """
        Initialize the KGGen instance and configure the language model.

        Args:
            - model: LLM identifier (e.g., "openai/gpt-4o"). Used to configure
                the provider-specific constraints and per-call parameters.
            - max_tokens: Maximum number of tokens the model can generate per call.
            - temperature: Sampling temperature to control output randomness.
            - reasoning_effort: Reasoning effort setting (e.g., "low", "medium",
                "high") if supported by the provider.
            - api_key: API key for the LLM provider. If None, environment or
                provider defaults are used.
            - api_base: Base URL for the provider API.
            - retrieval_model: SentenceTransformer model name used for retrieval.
            - disable_cache: Reserved for compatibility; LiteLLM manages caching
                externally if enabled by the provider.

        Returns:
            None.
        """
        # Store configuration values.
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.disable_cache = disable_cache
        self._usage_history: list[dict] = []

        # Initialize the model with the provided configuration.
        self.init_model(
            model=model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            retrieval_model=retrieval_model,
        )

    def validate_temperature(self, temperature: float):
        """
        Validate temperature against constraints of the current model.

        Args:
            - temperature: Temperature value to validate.

        Returns:
            None.

        Raises:
            ValueError: If the current model is in the gpt-5 family and the
                temperature is lower than 1.0.
        """
        # gpt-5 family requires temperature >= 1.0.
        if "gpt-5" in self.model and temperature < 1.0:
            raise ValueError("Temperature must be 1.0 for gpt-5 family models")

    def validate_max_tokens(self, max_tokens: int):
        """
        Validate max token limit against constraints of the current model.

        Args:
            - max_tokens: Maximum token count to validate.

        Returns:
            None.

        Raises:
            ValueError: If the current model is in the gpt-5 family and
                max_tokens is lower than 16000.
        """
        # gpt-5 family requires large context windows by default.
        if "gpt-5" in self.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        max_tokens: int = None,
        temperature: float = None,
        retrieval_model: str = None,
        api_key: str = None,
        api_base: str = None,
    ):
        """
        Initialize or reinitialize the LLM configuration with updated parameters.

        Args:
            - model: New LLM identifier. If None, keeps the current value.
            - reasoning_effort: New reasoning effort setting. If None, keeps
                the current value.
            - max_tokens: New maximum token limit. If None, keeps current value.
            - temperature: New sampling temperature. If None, keeps current value.
            - retrieval_model: Embedding model name for retrieval. When provided,
                a SentenceTransformer instance is created and stored.
            - api_key: New API key for the provider.
            - api_base: New base URL for the provider.

        Returns:
            None. Updates instance fields and cached configuration.

        Raises:
            ValueError: If temperature or max token constraints are violated.
        """
        # Update instance variables if new values were provided.
        if model is not None:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        if retrieval_model is not None:
            # Instantiate the retrieval model once to reuse for embeddings.
            self.retrieval_model = SentenceTransformer(retrieval_model)

        # Validate configuration before using the settings.
        self.validate_temperature(self.temperature)
        self.validate_max_tokens(self.max_tokens)

        # LiteLLM is configured per-call, so no client instance is created here.

    @staticmethod
    def from_file(file_path: str) -> Graph:
        """
        Load a Graph from a JSON file on disk.

        Args:
            - file_path: Path to a JSON file containing a dict compatible with
                the Graph constructor.

        Returns:
            Graph: Graph instance constructed from the JSON payload.

        Raises:
            OSError: If the file is not readable or the path is invalid.
            json.JSONDecodeError: If the file content is not valid JSON.
            TypeError: If the JSON structure does not match Graph fields.
        """
        # Read JSON from disk and construct the Graph instance.
        with open(file_path, "r") as f:
            graph = Graph(**json.load(f))
        return graph

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        """
        Construct a Graph from an in-memory dictionary.

        Args:
            - graph_dict: Dictionary with keys expected by Graph (e.g., entities,
                relations, edges, etc.).

        Returns:
            Graph: Graph instance constructed from the dictionary.

        Raises:
            TypeError: If the dictionary does not contain expected fields.
        """
        # Build a Graph directly from the provided mapping.
        return Graph(**graph_dict)
    
    @staticmethod
    def _parse_deduplication_method(
        method: DeduplicateMethod | str | None,
    ) -> DeduplicateMethod | None:
        """
        Normalize deduplication method input to DeduplicateMethod enum.
        """
        if method is None:
            return None
        if isinstance(method, DeduplicateMethod):
            return method
        if isinstance(method, str):
            normalized = method.strip().lower()
            for enum_value in DeduplicateMethod:
                if normalized == enum_value.value:
                    return enum_value
            raise ValueError(
                f"Unknown deduplication method '{method}'. "
                f"Valid values: {[m.value for m in DeduplicateMethod]}"
            )
        raise TypeError(
            "deduplication_method must be a DeduplicateMethod, string, or None"
        )

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        reasoning_effort: str = None,
        deduplication_method: DeduplicateMethod | str | None = DeduplicateMethod.SEMHASH,
        temperature: float = None,
        output_folder: Optional[str] = None,
    ) -> Graph:
        """
        Generate a knowledge graph from text or a conversation.

        Args:
            - input_data: Raw text or a list of message dicts with "role" and
                "content". Only "user" and "assistant" roles are included.
            - model: LLM name for extraction. If None, uses the instance value.
            - api_key: API key override for this call. If None, keeps instance value.
            - api_base: API base override for this call. If None, keeps instance value.
            - context: Optional data context description passed to deduplication.
            - chunk_size: Maximum chunk size (characters). If None, attempts a
                single pass and auto-falls back to 16384 on context length errors.
            - reasoning_effort: Reasoning effort override for this call.
            - deduplication_method: Deduplication strategy to apply after extraction.
                If None, deduplication is skipped.
            - temperature: Sampling temperature override for this call.
            - output_folder: Folder to save `graph.json`. If None, no file is saved.
        Returns:
            Graph: Knowledge graph containing extracted entities and relations.

        Raises:
            ValueError: If conversation messages are not dicts with "role"
                and "content".
            Exception: Propagates extraction errors except for "context length",
                which triggers automatic chunking.
        """

        def _process(content):
            """
            Extract entities and relations from a single content chunk.

            Args:
                content: Text (or chunk) to process.

            Returns:
                tuple:
                    - set[str]: Extracted entities.
                    - set[tuple[str, str, str]]: Extracted relations.
            """
            # Extract entities first to guide relation extraction.
            entities_with_reasoning = get_entities(
                content,
                is_conversation,
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                usage_history=self._usage_history,
            )

            # extract only names
            entities = [name for name, reasoning in entities_with_reasoning]

            # Extract relations conditioned on entities.
            relations = get_relations(
                content,
                entities,
                is_conversation=is_conversation,
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                usage_history=self._usage_history,
            )

            return entities_with_reasoning, relations
            
        # Determine input mode (conversation vs. raw text).
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            # Normalize messages into a single text string.
            text_content = []
            for message in input_data:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys"
                    )
                if message["role"] in ["user", "assistant"]:
                    # Preserve role tags to help the model parse dialogue.
                    text_content.append(f"{message['role']}: {message['content']}")

            # Join with newlines to preserve message boundaries.
            processed_input = "\n".join(text_content)
        else:
            # Use raw input directly.
            processed_input = input_data

        # Reinitialize configuration if any runtime overrides were provided.
        if any([model, temperature, api_key, api_base, reasoning_effort]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
                reasoning_effort=reasoning_effort or self.reasoning_effort,
            )
        
        entities_with_reasoning:List[tuple[str,str]] = []
        relations:List[tuple[str, str, str]] = []
        # If no chunk size is provided, attempt a single pass.
        if not chunk_size:
            try:
                entities_with_reasoning, relations = _process(processed_input)
            except Exception as e:
                # If the model reports a context length issue, enable chunking.
                if "context length" in str(e).lower():
                    print(
                        f"Context length error: {e}. Chunking text with chunk size 16384."
                    )
                    chunk_size = 16384
                else:
                    # Bubble up unexpected errors.
                    raise e

        if chunk_size:
            # Split the input into chunks to fit model context limits.
            chunks = chunk_text(processed_input, chunk_size)
            entities_with_reasoning = []
            relations = []

            # Process chunks in parallel to speed up extraction.
            with ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(_process, chunk): chunk for chunk in chunks
                }

                for future in as_completed(future_to_chunk):
                    # Merge per-chunk results into global sets.
                    chunk_entities_with_reasoning, chunk_relations = future.result()
                    entities_with_reasoning.extend(chunk_entities_with_reasoning)
                    relations.update(chunk_relations)

        # Build the Graph object from extracted entities and relations.
        entities_dict = {name: reasoning for name, reasoning in entities_with_reasoning}
        graph = Graph(
            entities=entities_dict,
            relations=relations,
            edges={r[1] for r in relations},
        )

        deduplication_method = self._parse_deduplication_method(deduplication_method)
        if deduplication_method:
            # Deduplicate the graph if a method is provided.
            graph = self.deduplicate(
                graph, method=deduplication_method, context=context
            )

        if output_folder:
            # Persist output if a folder is specified.
            self.export_graph(graph, os.path.join(output_folder, "graph.json"))
        return graph

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod | str = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
    ) -> Graph:
        """
        Deduplicate entities and relations in a graph using a specific method.

        Args:
            - graph: Input graph to deduplicate.
            - method: Deduplication strategy to use (e.g., FULL, SEMHASH).
            - semhash_similarity_threshold: Similarity threshold for SEMHASH;
                higher values are more conservative (recommended to keep at 0.95)
            - model: LLM name for deduplication. If None, uses instance model.
            - temperature: Sampling temperature for deduplication. If None,
                uses instance value.
            - api_key: API key override. If None, uses instance value.
            - api_base: API base override. If None, uses instance value.
            - context: Optional context description passed to the deduplication
                algorithm if supported.

        Returns:
            Graph: Deduplicated graph with cluster metadata when applicable.
        """
        # Normalize method to enum and reinitialize config if needed.
        method = self._parse_deduplication_method(method)
        # Reinitialize configuration with runtime overrides if provided.
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        # Build argument payload for deduplication.
        run_kwargs = {
            "graph": graph,
            "method": method,
            "retrieval_model": self.retrieval_model,
            "semhash_similarity_threshold": semhash_similarity_threshold,
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "usage_history": self._usage_history,
        }
        if context:
            # Only include context when explicitly provided.
            run_kwargs["context"] = context
        # Execute deduplication and return the new graph.
        return run_deduplication(**run_kwargs)

    def aggregate(self, graphs: list[Graph]) -> Graph:
        """
        Aggregate multiple Graph instances into a single combined graph.

        Maintains separate entities for each Step (Goal, Intention, Statement, etc.) 
        even if they have the same type, preventing unintended merging of duplicates.

        Adds the following semantic connections automatically:
            - trace:nextStep between consecutive Steps
            - trace:hasStep from Run to all Steps
            - trace:endsWith from Run to TerminationSignals

        Args:
            graphs (list[Graph]): List of Graph instances to merge.

        Returns:
            Graph: Aggregated graph containing all entities, relations, edges, 
                and entity metadata.
        """
        # Initialize containers for the aggregated graph
        all_entities: dict[str, str] = {}
        all_relations: set[tuple[str, str, str]] = set()
        all_edges: set[str] = set()
        all_entity_metadata: dict[str, set[str]] = {}

        steps_ordered: list[str] = []
        termination_nodes: list[str] = []
        run_node: str | None = None
        
        # Iterate over all graphs to merge their content
        for g_index, graph in enumerate(graphs):
            entity_id_map = {}

            # Add entities with unique IDs to avoid collisions across graphs
            for entity_id, entity_type in graph.entities.items():
                unique_id = f"{entity_id}_G{g_index}"  # ID univoco
                all_entities[unique_id] = entity_type
                entity_id_map[entity_id] = unique_id

                # Copy metadata if available
                if graph.entity_metadata and entity_id in graph.entity_metadata:
                    all_entity_metadata[unique_id] = graph.entity_metadata[entity_id].copy()

            # Update relations with unique entity IDs
            for subj, rel, obj in graph.relations:
                all_relations.add((entity_id_map.get(subj, subj), rel, entity_id_map.get(obj, obj)))
                all_edges.add(rel)

            # Collect Step and TerminationSignal nodes using unique IDs
            steps_in_graph = [entity_id_map[e] for e, t in graph.entities.items() if e.startswith("trace:Step")]
            steps_ordered.extend(steps_in_graph)

            term_nodes = [entity_id_map[e] for e, t in graph.entities.items() if "TerminationSignal" in t]
            termination_nodes.extend(term_nodes)

            # Identify Run node (take the first one found)
            if not run_node:
                run_candidates = [entity_id_map[e] for e, t in graph.entities.items() if e.startswith("trace:Run")]
                if run_candidates:
                    run_node = run_candidates[0]

        # Sort Steps by numeric ID to preserve reasoning order
        def step_sort_key(step_name: str):
            try:
                return int(step_name.split("_")[1])
            except:
                return 0

        steps_ordered = sorted(steps_ordered, key=step_sort_key)

        # Connect consecutive steps with trace:nextStep
        for i in range(len(steps_ordered) - 1):
            all_relations.add((steps_ordered[i], "trace:nextStep", steps_ordered[i + 1]))
            all_edges.add("trace:nextStep")

        # Connect all Steps to the Run node
        if run_node:
            for step in steps_ordered:
                all_relations.add((run_node, "trace:hasStep", step))
                all_edges.add("trace:hasStep")

            # Connect Run node to TerminationSignal nodes
            for term in termination_nodes:
                all_relations.add((run_node, "trace:endsWith", term))
                all_edges.add("trace:endsWith")
        
        # Return the aggregated graph
        return Graph(
            entities=all_entities,
            relations=all_relations,
            edges=all_edges,
            entity_metadata=all_entity_metadata if all_entity_metadata else None
        )

    @staticmethod
    def visualize(graph: Graph, output_path: str, open_in_browser: bool = False):
        """
        Generate a visualization for a graph and save it to a file.

        Args:
            graph: Graph to visualize.
            output_path: Output path for the visualization file (e.g., HTML).
            open_in_browser: If True, opens the output file in a browser.

        Returns:
            None.
        """
        # Delegate to the visualization utility.
        visualize_kg(graph, output_path, open_in_browser=open_in_browser)

    # ====== Token Usage ======
    def reset_token_usage(self):
        """
        Reset token usage history.

        Args:
            None.

        Returns:
            None.
        """
        # Clear the local usage history.
        self._usage_history = []

    def extract_token_usage_from_history(self) -> Dict[str, int]:
        """
        Extract token usage statistics from local history.

        Args:
            None.

        Returns:
            Dict[str, int]: Dictionary with keys:
                - "prompt_tokens": Total prompt token count.
                - "completion_tokens": Total completion token count.
                - "total_tokens": Total token count.
        """

        # Initialize counters for aggregation.
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in self._usage_history:
            if isinstance(entry, dict):
                # Check for usage information in various possible locations.
                usage = entry.get("usage") or entry.get("response", {}).get("usage")

                if usage:
                    # Accumulate usage counts safely with defaults.
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

        # Return an aggregated usage summary.
        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        }

    # ====== Retrieval Methods ======

    # def _parse_embedding_model(
    #     self, model: Optional[SentenceTransformer] = None
    # ) -> Optional[SentenceTransformer]:
    #     """
    #     Resolve the embedding model to use for retrieval operations.

    #     Args:
    #         - model: Optional SentenceTransformer instance. If None, uses
    #             `self.retrieval_model`.

    #     Returns:
    #         SentenceTransformer: Ready-to-use embedding model.

    #     Raises:
    #         ValueError: If no embedding model is available.
    #     """
    #     # Prefer the provided model, otherwise fall back to instance config.
    #     if model is None:
    #         model = self.retrieval_model
    #     if model is None:
    #         raise ValueError("No retrieval model provided")
    #     return model

    # @staticmethod
    # def to_nx(graph: Graph) -> nx.DiGraph:
    #     """
    #     Convert a Graph into a NetworkX directed graph.

    #     Args:
    #         graph: Input Graph to convert.

    #     Returns:
    #         nx.DiGraph: Directed graph with nodes = entities and edges = relations.
    #     """
    #     # Initialize an empty directed graph.
    #     G = nx.DiGraph()
    #     # Add all entities as nodes.
    #     for entity in graph.entities:
    #         G.add_node(entity)

    #     # Add directed edges with relation labels.
    #     for relation in graph.relations:
    #         source, rel, target = relation
    #         G.add_edge(source, target, relation=rel)
    #     return G

    # def generate_embeddings(
    #     self,
    #     graph: Union[Graph, nx.DiGraph],
    #     model: Optional[SentenceTransformer] = None,
    # ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    #     """
    #     Generate embeddings for nodes and relations of a graph.

    #     Args:
    #         graph: Graph or nx.DiGraph. If Graph, it is converted to NetworkX.
    #         model: Embedding model to use. If None, uses instance retrieval model.

    #     Returns:
    #         tuple:
    #             - dict[str, np.ndarray]: Node -> embedding map.
    #             - dict[str, np.ndarray]: Relation -> embedding map.
    #     """
    #     # Resolve the embedding model.
    #     model = self._parse_embedding_model(model)
    #     if isinstance(graph, Graph):
    #         # Convert to NetworkX to simplify iteration.
    #         graph = self.to_nx(graph)

    #     # Encode each node string into an embedding vector.
    #     node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
    #     relation_embeddings = {
    #         rel: model.encode(rel).tolist()
    #         # TODO: this is triggering index out of range error
    #         for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
    #     }
    #     return node_embeddings, relation_embeddings

    # def retrieve(
    #     self,
    #     query: str,
    #     node_embeddings: dict[str, np.ndarray],
    #     graph: nx.DiGraph,
    #     model: Optional[SentenceTransformer] = None,
    #     k: int = 8,
    #     verbose: bool = False,
    # ) -> tuple[list[tuple[str, float]], set[str], str]:
    #     """
    #     Retrieve top-k relevant nodes and build a textual context for a query.

    #     Args:
    #         query: Query text.
    #         node_embeddings: Precomputed node -> embedding map.
    #         graph: NetworkX graph used to reconstruct local context.
    #         model: Embedding model for the query. If None, uses instance retrieval model.
    #         k: Number of top similar nodes to return.
    #         verbose: If True, prints per-node context and the final combined context.

    #     Returns:
    #         tuple:
    #             - list[tuple[str, float]]: Top-k nodes with similarity scores.
    #             - set[str]: Context sentences extracted from the graph.
    #             - str: Combined context string (space-joined sentences).
    #     """
    #     # Resolve the embedding model for the query.
    #     model = self._parse_embedding_model(model)
    #     # Find the most relevant nodes by cosine similarity.
    #     top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
    #     context = set()
    #     for node, _ in top_nodes:
    #         # Collect context around each top node.
    #         node_context = self.retrieve_context(node, graph)
    #         if verbose:
    #             print(f"Context for node {node}: {node_context}")
    #         context.update(node_context)
    #     # Join context sentences into a single text block.
    #     context_text = " ".join(context)
    #     if verbose:
    #         print(f"Combined context: '{context_text}'\n---")
    #     return top_nodes, context, context_text

    # @staticmethod
    # def retrieve_relevant_nodes(
    #     query: str,
    #     node_embeddings: dict[str, np.ndarray],
    #     model: SentenceTransformer,
    #     k: int = 8,
    # ) -> list[tuple[str, float]]:
    #     """
    #     Compute similarity between the query and all nodes and return top-k.

    #     Args:
    #         query: Query text.
    #         node_embeddings: Node -> embedding map.
    #         model: Embedding model used to encode the query.
    #         k: Maximum number of nodes to return.

    #     Returns:
    #         list[tuple[str, float]]: Sorted list of (node, similarity) in
    #             descending similarity order.
    #     """
    #     # Encode the query once for efficiency.
    #     query_embedding = model.encode(query).reshape(1, -1)
    #     similarities = []
    #     for node, embed in node_embeddings.items():
    #         # Compute cosine similarity with each node embedding.
    #         target_embedding = np.array(embed).reshape(1, -1)
    #         similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
    #         similarities.append((node, similarity))
    #     # Sort by similarity and return top-k.
    #     similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    #     return similarities[:k]

    # @staticmethod
    # def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2) -> list[str]:
    #     """
    #     Extract local context for a node by exploring incoming and outgoing neighbors.

    #     Args:
    #         node: Starting node.
    #         graph: NetworkX graph to explore.
    #         depth: Maximum traversal depth (>= 1).

    #     Returns:
    #         list[str]: Sentences describing relations involving the node and
    #             its neighbors up to the specified depth.
    #     """
    #     # Use a set to avoid duplicate context sentences.
    #     context = set()

    #     def explore_neighbors(current_node, current_depth):
    #         """
    #         Recursively explore incoming and outgoing neighbors of a node.

    #         Args:
    #             current_node: Node currently being expanded.
    #             current_depth: Current recursion depth.

    #         Returns:
    #             None. Adds descriptive sentences to `context`.
    #         """
    #         # Stop recursion if depth limit is exceeded.
    #         if current_depth > depth:
    #             return
    #         # Outgoing edges.
    #         for neighbor in graph.neighbors(current_node):
    #             rel = graph[current_node][neighbor]["relation"]
    #             context.add(f"{current_node} {rel} {neighbor}.")
    #             explore_neighbors(neighbor, current_depth + 1)
    #         # Incoming edges.
    #         for neighbor in graph.predecessors(current_node):
    #             rel = graph[neighbor][current_node]["relation"]
    #             context.add(f"{neighbor} {rel} {current_node}.")
    #             explore_neighbors(neighbor, current_depth + 1)

    #     # Start traversal from the given node.
    #     explore_neighbors(node, 1)
    #     return list(context)

    # @staticmethod
    # def export_graph(graph: Graph, output_path: str):
    #     """
    #     Export a Graph to JSON on disk.

    #     Args:
    #         graph: Graph to export.
    #         output_path: Full path of the output JSON file.

    #     Returns:
    #         None.

    #     Raises:
    #         OSError: If the directory cannot be created or the file cannot be written.
    #     """
    #     # Ensure the output directory exists.
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     # Build a serializable dictionary payload.
    #     graph_dict = {
    #         "entities": list(graph.entities),
    #         "relations": list(graph.relations),
    #         "edges": list(graph.edges),
    #         "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
    #         if graph.entity_clusters
    #         else None,
    #         "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
    #         if graph.edge_clusters
    #         else None,
    #         "entity_metadata": graph.entity_metadata,
    #     }

    #     # Write JSON to disk.
    #     with open(output_path, "w") as f:
    #         json.dump(graph_dict, f, indent=2)
