"""
compute_global_metrics.py
-------------------------
Calcola metriche globali dai Knowledge Graph delle run.

Metriche:
- Density
- Branching factor (Statements per Step, Intentions per Goal)
"""

import os
import glob
import json
import networkx as nx
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ===========================
# UTILS
# ===========================

def extract_type(node_name: str):
    """
    Estrae il tipo semantico del nodo dal nome RDF-like.
    Esempio: mind:Goal_1_G0 -> Goal
    """
    try:
        after_namespace = node_name.split(":")[1]
        node_type = after_namespace.split("_")[0]
        return node_type
    except:
        return "Unknown"

def build_graph(json_path):
    """
    Costruisce un grafo NetworkX da un JSON con nodi come stringhe.
    """
    with open(json_path) as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Aggiungi nodi con tipo
    for node in data["nodes"]:
        node_type = extract_type(node)
        G.add_node(node, type=node_type)

    # Aggiungi archi con predicate
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            predicate=edge.get("predicate")
        )

    return G

# ===========================
# METRICHE GLOBALI
# ===========================

def compute_global_metrics(G):
    """
    Calcola le metriche globali richieste sul grafo.
    """
    metrics = {}

    # --- DENSITY ---
    # Misura quanto il grafo è connesso rispetto al massimo numero possibile di archi
    # density = |E| / (|V| * (|V|-1))
    # Indica quanto sono interconnessi gli elementi cognitivi, grafi troppo densi potrebbero indicare
    # un reasoning eccessivo, ridondante, troppo sparsi potrebbero invece indicare mancanze di collegamenti logici
    metrics["density"] = nx.density(G)

    # --- BRANCHING FACTOR ---
    # Due metriche legate alla ramificazione dei nodi

    # Statements per Step
    # Misura quanti statements contribuiscon mediamente a ciascun Step
    # - Un valore alto indica che gli step sono supportati da molte informazioni (potenziale reasoning complesso)
    # - Un valore basso può indicare steps poco motivati
    steps = [n for n,d in G.nodes(data=True) if d["type"]=="Step"]
    step_branching = []
    for step in steps:
        stmts = [p for p in G.predecessors(step) if G.nodes[p]["type"]=="Statement"]
        step_branching.append(len(stmts))
    metrics["avg_statements_per_step"] = sum(step_branching)/len(step_branching) if step_branching else 0

    # Intentions per Goal
    # Numero medio di Intention associate a ciascun Goal
    goals = [n for n,d in G.nodes(data=True) if d["type"]=="Goal"]
    goal_branching = []
    for goal in goals:
        intents = [s for s in G.successors(goal) if G.nodes[s]["type"]=="Intention"]
        goal_branching.append(len(intents))
    metrics["avg_intentions_per_goal"] = sum(goal_branching)/len(goal_branching) if goal_branching else 0

    return metrics

# ===========================
# PIPELINE
# ===========================

def process_folder(input_folder, output_csv="global_metrics.csv"):
    """
    Calcola metriche per tutti i file JSON in una cartella.
    """
    all_rows = []

    json_files = glob.glob(os.path.join(input_folder, "*.json"))

    for file_path in json_files:
        with open(file_path) as f:
            data = json.load(f)

        G = build_graph(file_path)
        metrics = compute_global_metrics(G)

        row = {
            "run_id": os.path.basename(file_path).replace(".json",""),
            "label": data.get("label", "unknown")  # ora 'data' esiste
        }
        row.update(metrics)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.fillna(0, inplace=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / output_csv
    df.to_csv(output_path, index=False)
    print(f"[INFO] Global metrics saved to {output_path}")

# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "json_graphs")
    process_folder(input_folder, output_csv="global_metrics.csv")