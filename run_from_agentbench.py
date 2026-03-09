import os
import time
import json
import traceback
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from kg_gen import KGGen
from litellm.exceptions import RateLimitError
from pydantic_core import ValidationError
from kg_gen.input_generation.core.base_loader import load_failed_samples, save_failed_jsonl, build_reasonings_structure, extract_model_from_config
from kg_gen.input_generation.tasks.dbbench import DBBenchExtractor
from kg_gen.input_generation.tasks.os_interaction import OSInteractionExtractor


# Load environment variables
load_dotenv()


def pretty_print_graph(graph, title="Graph"):
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    entities = graph.entities or {}
    edges = graph.edges or set()
    relations = graph.relations or set()

    print("\nEntities:")
    if entities:
        for key in sorted(entities.keys()):
            print(f"  - {key}:")
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


def get_sample_label(sample):
    """
    Genera una label stringa per un sample, combinando `status` e `result`.
    Esempio: "completed_true", "agent invalid action_false", ecc.
    """
    try:
        output = sample.get("output", {})
        status = output.get("status", "unknown").replace(" ", "_")
        result = output.get("result", {}).get("result", "unknown")
        label = f"{status}_{result}"
    except Exception as e:
        print(f"Error generating label for sample {sample.get('index')}: {e}")
        label = "unknown_unknown"
    return label


# Per salvare i componenti (nodi e archi) del grafo sotto forma di json ordinato
def save_graph_as_json(graph, output_path, sample=None):
    """
    Salva il grafo in un JSON con la struttura:
    {
        "nodes": [...],
        "edges": [
            {"source": "...", "target": "...", "predicate": "..."},
        ]
    }
    """
    data = {
        "nodes": list(graph.entities.keys()) if graph.entities else [],
        "edges": []
    }

    if graph.relations:
        for subj, pred, obj in graph.relations:
            data["edges"].append({
                "source": subj,
                "target": obj,
                "predicate": pred
            })
    
    # Aggiungi la label se il sample è fornito
    if sample:
        data["label"] = get_sample_label(sample)

    # Salva il JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved graph JSON to: {output_path}")



def safe_generate(kg, reasoning_block, context, max_retries=10):
    for attempt in range(max_retries):
        try:
            return kg.generate(input_data=reasoning_block, context=context)
        except RateLimitError as e:
            wait_time = 2 * (attempt + 1)
            print(f"[RateLimit] Sleeping {wait_time}s...")
            print(f"Error details: {e}")
            time.sleep(wait_time)
        except ValidationError as e:
            wait_time = 2 * (attempt + 1)
            print("[Invalid JSON] Model returned truncated output.")
            print(f"Error details: {e}")
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            wait_time = 2 * (attempt + 1)
            print(f"[Other Error] Attempt {attempt+1}/{max_retries}")
            traceback.print_exc()
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded.")



def get_extractor(task_type):
    if task_type.startswith("dbbench"):
        return DBBenchExtractor()
    elif task_type.startswith("os"):
        return OSInteractionExtractor()
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# generazione dei grafi per un sample (un task)
def generate_graphs_for_sample(kg, sample, extractor, model_name, graphs_root_dir, context):
    sample_index = sample.get("index", "unknown")
    safe_sample_id = str(sample_index).replace("/", "_").replace(" ", "_")
    sample_dir = graphs_root_dir / f"aggregated_graph_{safe_sample_id}"
    sample_dir.mkdir(exist_ok=True)

    # Build reasoning steps
    reasoning_blocks = build_reasonings_structure(sample, model_name=model_name, extractor=extractor)

    # Save KGGen input for traceability
    input_file = sample_dir / "kggen_input.json"
    if not input_file.exists():
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(reasoning_blocks, f, indent=2, ensure_ascii=False)

    # Collect graphs, skip already generated steps
    graphs = {}
    for step_id, block in reasoning_blocks.items():
        graph_html_path = sample_dir / f"step_{step_id}.html"
        if graph_html_path.exists():
            continue

        print(f"Generating graph for step {step_id}...")
        graph = safe_generate(kg, block, context)
        graphs[step_id] = graph

        # Save per-step visualization
        kg.visualize(graph, str(graph_html_path), open_in_browser=False)

    # Aggregate all graphs
    all_graphs = [graphs[step_id] for step_id in sorted(graphs.keys())]
    if all_graphs:
        output_html = sample_dir / "aggregated_graph.html"
        if not output_html.exists():
            print("Generating aggregated graph...")
            aggregated_graph = kg.aggregate(all_graphs)
            kg.visualize(aggregated_graph, str(output_html), open_in_browser=False)
            print(f"Saved aggregated graph to: {output_html}")

            # Salva JSON con nodi e archi

            # Cartella di destinazione dei JSON
            base_dir = Path(__file__).parent  # kg-gen-update
            json_graphs_dir = base_dir / "kg_gen" / "error_signature_discovery" / "json_graphs"
            json_graphs_dir.mkdir(parents=True, exist_ok=True)

            # Salva JSON con nodi e archi usando l'index del sample
            sample_index = sample.get("index", "unknown")
            safe_sample_id = str(sample_index).replace("/", "_").replace(" ", "_")
            json_output_path = json_graphs_dir / f"aggregated_graph_{safe_sample_id}.json"

            save_graph_as_json(aggregated_graph, json_output_path, sample=sample)


def process_all_samples(results_root, task_name, only_failed:bool):
    """
    Genera grafi per tutti i sample (solo failed o tutti) di un task specifico.
    """
    jsonl_path = results_root / "qwen3-14b-ollama" / task_name / "runs.jsonl"
    if not jsonl_path.exists():
        print(f"No runs.jsonl found for {task_name}, skipping...")
        return
    
    samples = []

    # se si desidera estrarre i grafi solo per i failed samples, si estraggono (se già non sono stati estratti) e si ricavano i samples
    if only_failed:
        # se i failed samples erano già stati filtrati, non viene ripetuta l'operazione
        failed_samples_files = list(results_root.glob(f"{task_name}_failed_samples_*.jsonl"))
        if failed_samples_files:
            print(f"Failed samples already extracted for {task_name}, loading the latest...")
            latest_failed_file = sorted(failed_samples_files)[-1]
            with open(latest_failed_file, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f]
        else:
            samples = load_failed_samples(jsonl_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_output_path = results_root / f"{task_name}_failed_samples_{timestamp}.jsonl"
            save_failed_jsonl(samples, failed_output_path)
            print(f"Saved failed samples to: {failed_output_path}")

        if not samples:
            print("No failed samples found. Skipping...")
            return
    # altrimenti consideriamo tutti i samples dei task corretti e non
    else:
        with open(jsonl_path, 'r', encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]

    # Inizializza KGGen
    kg = KGGen(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        api_base=os.getenv("API_BASE"),
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )
    context = "AgentBench Reasoning"
    extractor = get_extractor(task_name)

    graphs_root_dir = Path(results_root).parent / f"graphs_{results_root.name}_{task_name}"
    graphs_root_dir.mkdir(exist_ok=True)

    for sample in samples:
        generate_graphs_for_sample(kg, sample, extractor, extract_model_from_config(results_root / "config.yaml"), graphs_root_dir, context)


def process_single_sample(results_root, task_name, sample_index):
    """
    Genera grafi solo per uno specifico sample dato l'index.
    """
    jsonl_path = results_root / "qwen3-14b-ollama" / task_name / "runs.jsonl"
    if not jsonl_path.exists():
        print(f"No runs.jsonl found for {task_name}, skipping...")
        return

    if jsonl_path:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
    
    sample = next((s for s in samples if str(s.get("index")) == str(sample_index)), None)
    if not sample:
        print(f"No sample with index {sample_index} found. Skipping...")
        return

    kg = KGGen(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        api_base=os.getenv("API_BASE"),
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )

    context = "AgentBench Reasoning"
    extractor = get_extractor(task_name)
    graphs_root_dir = Path(results_root).parent / f"graphs_{results_root.name}_{task_name}"
    graphs_root_dir.mkdir(exist_ok=True)

    generate_graphs_for_sample(kg, sample, extractor, extract_model_from_config(results_root / "config.yaml"), graphs_root_dir, context)


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    results_root = base_dir / "kg_gen" / "input_generation" / "agentbench_results" / "2026-02-25-15-20-17"
    
    process_single_sample(results_root, "os-std", "std-001-stock-00000")

    
