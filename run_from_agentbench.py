import os
import time
from dotenv import load_dotenv
from pathlib import Path
from kg_gen import KGGen
import traceback
from kg_gen.input_generation.kggen_input_os_interaction import get_failed_samples_inputs, load_failed_samples
import litellm
from litellm.exceptions import RateLimitError
import json
from pydantic_core import ValidationError


# Load environment variables
load_dotenv()


def pretty_print_graph(graph, title="Graph"):
    """
    Pretty-print the components of a knowledge graph.

    This function prints entities, edges, relations, and optional clusters
    in a structured and human-readable format.

    Args:
        graph: A KGGen graph object (Pydantic-based).
        title (str): Title displayed before the graph content.
    """
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)

    entities = graph.entities or {}
    edges = graph.edges or set()
    relations = graph.relations or set()

    print("\nEntities:")
    if entities:
        for key in sorted(entities.keys()):
            print(f"\n  - {key}:")
            value = entities[key]
            if isinstance(value, dict):
                print(json.dumps(value, indent=6))
            else:
                print(f"      {value}")
    else:
        print("  (none)")

    print("\nEdges:")
    if edges:
        for edge in sorted(edges):
            print(f"  - {edge}")
    else:
        print("  (none)")

    print("\nRelations:")
    if relations:
        for rel in sorted(relations):
            print(f"  - {rel[0]} -- {rel[1]} --> {rel[2]}")
    else:
        print("  (none)")

    if graph.entity_clusters:
        print("\nEntity Clusters:")
        for cluster, members in graph.entity_clusters.items():
            print(f"  - {cluster}: {members}")

    if graph.edge_clusters:
        print("\nEdge Clusters:")
        for cluster, members in graph.edge_clusters.items():
            print(f"  - {cluster}: {members}")

    print("="*60 + "\n")


def safe_generate(kg, reasoning_block, context, max_retries=10):
    """
    Safely generate a knowledge graph from a reasoning block.

    Handles RateLimitError, ValidationError, and other exceptions
    with exponential backoff retries. Prints full error details for
    debugging.

    Args:
        kg: KGGen instance
        reasoning_block: Input data for a single reasoning step
        context: Context string passed to KGGen
        max_retries: Maximum number of retry attempts on error

    Returns:
        Generated KGGen graph object
    """
    for attempt in range(max_retries):
        try:
            return kg.generate(input_data=reasoning_block, context=context)

        except RateLimitError as e:
            # Handle API rate limiting
            wait_time = 2 * (attempt + 1)
            print(f"[RateLimit] Sleeping {wait_time:.2f}s...")
            print(f"Error details: {e}")
            time.sleep(wait_time)

        except ValidationError as e:
            # Handle invalid JSON returned by the model
            wait_time = 2 * (attempt + 1)
            print("[Invalid JSON] Model returned truncated output.")
            print(f"Error details: {e}")
            print(f"Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

        except Exception as e:
            # Catch-all for other unexpected errors
            wait_time = 2 * (attempt + 1)
            print(f"[Other Error] Attempt {attempt+1}/{max_retries}")
            print("Full traceback:")
            traceback.print_exc()  # <-- stampa tutto lo stack trace
            print(f"Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

    raise Exception("Max retries exceeded.")


if __name__ == "__main__":

    # --- Paths to AgentBench results ---
    base_dir = Path(__file__).parent
    results_root = base_dir / "kg_gen" / "input_generation" / "agentbench_results" / "2026-02-25-15-20-17"
    jsonl_path = results_root / "qwen3-14b-ollama-thinking-parameter" / "os-std" / "runs.jsonl"
    config_path = results_root / "config.yaml"

    # Folder name to store generated graphs
    results_folder_name = results_root.name
    graphs_root_dir = base_dir / f"graphs_{results_folder_name}"
    graphs_root_dir.mkdir(exist_ok=True)

    # --- Initialize KGGen ---
    kg = KGGen(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        api_base=os.getenv("API_BASE"),
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )

    context = "AgentBench Reasoning"

    print("Generating KGGen input from failed AgentBench samples...")
    kg_inputs_list = get_failed_samples_inputs(jsonl_path, config_path)

    if not kg_inputs_list:
        print("No failed samples found. Exiting.")
        exit()

    # Load all runs to retrieve real sample IDs
    with open(jsonl_path, "r", encoding="utf-8") as f:
        all_runs = [json.loads(line) for line in f]

    # Filter only failed runs
    failed_runs = load_failed_samples(jsonl_path)

    print(f"\n{len(kg_inputs_list)} failed samples available.")
    
    # Ask user to select mode
    mode = input(
        "Type:\n"
        "  - a  → generate graphs for ALL failed samples\n"
        "  - s  → select a single sample\n"
        "Choice: "
    ).strip().lower()

    # ============================================================
    # MODE: ALL FAILED SAMPLES
    # ============================================================
    if mode == "a":

        for idx, (run_data, selected_input) in enumerate(zip(failed_runs, kg_inputs_list)):

            # Retrieve the actual sample ID
            sample_id = run_data.get("index", f"sample-{idx:03d}")
            safe_sample_id = str(sample_id).replace("/", "_").replace(" ", "_")

            print(f"\n\n========== SAMPLE {safe_sample_id} ==========")

            # Create folder for this sample's graph
            sample_dir = graphs_root_dir / f"aggregated_graph_{safe_sample_id}"
            sample_dir.mkdir(exist_ok=True)

            # Save the input JSON used for KGGen
            input_file = sample_dir / "kggen_input.json"
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(selected_input, f, indent=2, ensure_ascii=False)

            graphs = {}
            
            # Generate graph for each reasoning step
            for step_id, reasoning_block in selected_input.items():
                print(f"Generating graph for reasoning step {step_id}...")
                graph = safe_generate(kg, reasoning_block, context)
                graphs[step_id] = graph
            
            print("Generating aggregated graph...")
            aggregated_graph = kg.aggregate(list(graphs.values()))

            # Save visualization HTML
            output_html = sample_dir / "aggregated_graph.html"
            kg.visualize(aggregated_graph, str(output_html), open_in_browser=False)

            print(f"Saved aggregated graph to: {output_html}")

        print("\nAll graphs generated successfully.")
        print(f"Output folder: {graphs_root_dir}")

    # ============================================================
    # MODE: SINGLE SAMPLE
    # ============================================================
    elif mode == "s":
        
        # Ask the user to enter the sample index
        sample_index_input = input("Enter the sample 'index' to process: ").strip()
        
        available_indices = [r.get("index") for r in failed_runs]

        if sample_index_input not in map(str, available_indices):
            print("Invalid sample index. Exiting.")
        exit()

        if not sample_idx.isdigit() or int(sample_idx) >= len(kg_inputs_list):
            print("Invalid choice. Exiting.")
            exit()

        # Find the corresponding entry in kg_inputs_list
        idx = next(i for i, r in enumerate(failed_runs) if str(r.get("index")) == sample_index_input)
        selected_input = kg_inputs_list[idx]

        # Safe sample ID for folder naming
        safe_sample_id = str(sample_index_input).replace("/", "_").replace(" ", "_")

        print(f"\n========== SAMPLE {safe_sample_id} ==========")
        
        # Create folder for graphs
        sample_dir = graphs_root_dir / f"aggregated_graph_{safe_sample_id}"
        sample_dir.mkdir(exist_ok=True)

        # Save KGGen input
        input_file = sample_dir / "kggen_input.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(selected_input, f, indent=2, ensure_ascii=False)

        graphs = {}
        
        # Generate and collect graphs for each reasoning step
        for step_id, reasoning_block in selected_input.items():
            print(f"\n=== Generating graph for reasoning step {step_id} ===")
            graph = safe_generate(kg, reasoning_block, context)
            graphs[step_id] = graph
        
        # Aggregate all step graphs
        aggregated_graph = kg.aggregate(list(graphs.values()))
        
        # Visualize and save
        output_html = sample_dir / "aggregated_graph.html"
        kg.visualize(aggregated_graph, str(output_html), open_in_browser=True)

        print(f"\nSaved aggregated graph to: {output_html}")
        print("Done.")

    else:
        print("Invalid option.")