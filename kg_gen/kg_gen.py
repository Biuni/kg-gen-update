import json
import os
import re
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
            entities = [name for name, _ in entities_with_reasoning]

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
        relations: set[tuple[str, str, str]] = set()

        # If no chunk size is provided, attempt a single pass.
        if not chunk_size:
            try:
                entities_with_reasoning, extracted_relations = _process(processed_input)
                relations = set(extracted_relations)
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
            relations = set()

            # Process chunks in parallel to speed up extraction.
            # with ThreadPoolExecutor() as executor:
            #     future_to_chunk = {
            #         executor.submit(_process, chunk): chunk for chunk in chunks
            #     }

            #     for future in as_completed(future_to_chunk):
            #         # Merge per-chunk results into global sets.
            #         chunk_entities, chunk_relations = future.result()
            #         entities_with_reasoning.extend(chunk_entities)
            #         relations.update(chunk_relations)
            
            # Process each chunk sequentially, extracting entities and relations
            for chunk in chunks:
                chunk_entities, chunk_relations = _process(chunk)
                entities_with_reasoning.extend(chunk_entities)
                relations.update(chunk_relations)

        # Build the Graph object from extracted entities and relations.
        entities_dict = {name: reasoning for name, reasoning in entities_with_reasoning}
        valid_entity_names = set(entities_dict.keys())

        relations = {
            (s, p, o)
            for (s, p, o) in relations
            if s in valid_entity_names and o in valid_entity_names
        }

        graph = Graph(
            entities=entities_dict,
            relations=relations,
            edges={r[1] for r in relations},
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
                if subj in entity_id_map and obj in entity_id_map:
                    new_rel = (
                        entity_id_map[subj],
                        rel,
                        entity_id_map[obj],
                    )
                    all_relations.add(new_rel)
                    all_edges.add(rel)

            # Collect Step and TerminationSignal nodes using unique IDs
            steps_in_graph = [
                entity_id_map[e]
                for e in graph.entities
                if e.startswith("trace:Step")
            ]
            steps_ordered.extend(steps_in_graph)

            term_nodes = [
                entity_id_map[e]
                for e in graph.entities
                if e.startswith("mind:TerminationSignal")
            ]
            termination_nodes.extend(term_nodes)

            # Identify Run node (take the first one found)
            if not run_node:
                run_candidates = [
                    entity_id_map[e]
                    for e in graph.entities
                    if e.startswith("trace:Run")
                ]
                if run_candidates:
                    run_node = run_candidates[0]

        # Sort Steps by numeric ID to preserve reasoning order
        def step_sort_key(step_name: str):
            match = re.search(r"S(\d+)", step_name)
            return int(match.group(1)) if match else 0

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

   