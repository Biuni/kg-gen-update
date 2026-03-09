"""
subgraph mining
------------------
Estrae motifs ricorrenti dai Knowledge Graph generati da ogni Run dell'agente (risultato ottenuto da un sample del task os-interaction
di AgentBench) e produce un conteggio di pattern cognitivi (Goal->Intention->ToolCall, ecc.).
Può essere usato come primo step nell'analisi di errori cognitivi.
"""

import networkx as nx
import json
from collections import defaultdict
import os
from pathlib import Path
import glob
import pandas as pd

# Funzione di caricamento JSON
def load_json(file_path):
    """Carica un JSON da file"""
    with open(file_path, "r") as f:
        return json.load(f)

# Funzione costruzione grafo
def build_graph(data):
    """Costruisce un grafo diretto NetworkX dal JSON"""
    G = nx.DiGraph()
    # Aggiungi nodi con tipo
    for n in data['nodes']:
        node_type = n.split(":")[1].split("_")[0]  # mind:Goal_1_G0 -> Goal
        G.add_node(n, type=node_type)
    # Aggiungi archi con predicato
    for e in data['edges']:
        source = e['source']
        target = e['target']
        predicate = e.get('predicate', None)
        G.add_edge(source, target, predicate=predicate)
    return G


# Funzione di conteggio sequenze significative
def count_motifs(G):
    """Conta le occorrenze dei motifs definiti ontologicamente"""
    counts = defaultdict(int)
    
    # ======================
    # Motifs di lunghezza 2
    # ======================
    
    # -------------------
    # 1. Statement -> Step
    # Verifica la presenza di statements connessi agli step e ne verifica la cardinalità
    for stmt in [n for n,d in G.nodes(data=True) if d['type']=='Statement']:
        for step in G.successors(stmt):
            if G.nodes[step]['type']=='Step' and G.edges[stmt,step]['predicate']=='mind:assertedIn':
                counts[("Statement","Step")] += 1

    # -------------------
    # 2. Intention -> Step
    # Verifica la presenza di intentions connesse agli step e ne verifica la cardinalità
    for intention in [n for n,d in G.nodes(data=True) if d['type']=='Intention']:
        for step in G.successors(intention):
            if G.nodes[step]['type']=='Step' and G.edges[intention,step]['predicate']=='mind:expressedIn':
                counts[("Intention","Step")] += 1

    # -------------------
    # 3. Goal -> Intention 
    # Pattern utile per verificare motivazioni senza arrivare alla ToolCall
    for goal in [n for n,d in G.nodes(data=True) if d['type']=='Goal']:
        for intention in G.successors(goal):
            if G.nodes[intention]['type']=='Intention' and G.edges[goal,intention]['predicate'] == 'mind:motivates':
                counts[("Goal","Intention")] += 1

    # -------------------
    # 4. Goal -> Step
    # Verifica che i Goal siano associati correttamente agli Steps
    for goal in [n for n,d in G.nodes(data=True) if d['type']=='Goal']:
        for step in G.successors(goal):
            if G.nodes[step]['type']=='Step' and G.edges[goal,step]['predicate']=='mind:goalInStep':
                counts[("Goal","Step")] += 1

    # -------------------
    # 5. Intention -> ToolCall
    # Verifica la presenza di intentions che giustifichino o portino a una tool call e ne verifica la cardinalità
    for intention in [n for n,d in G.nodes(data=True) if d['type']=='Intention']:
        for call in G.successors(intention):
            if G.nodes[call]['type']=='ToolCall' and G.edges[intention,call]['predicate']=='mind:commitsToToolCall':
                counts[("Intention","ToolCall")] += 1

    # -------------------
    # 6. ToolCall -> Step
    # Verifica la presenza di tool calls connesse agli step e ne verifica la cardinalità
    for call in [n for n,d in G.nodes(data=True) if d['type']=='ToolCall']:
        for step in G.successors(call):
            if G.nodes[step]['type']=='Step' and G.edges[call,step]['predicate']=='act:usedInStep':
                counts[("ToolCall","Step")] += 1

    # -------------------
    # 7. Observation -> Step
    # Verifica la presenza di observations connesse agli step e ne verifica la cardinalità
    for obs in [n for n,d in G.nodes(data=True) if d['type']=='Observation']:
        for step in G.successors(obs):
            if G.nodes[step]['type']=='Step' and G.edges[obs,step]['predicate']=='obs:observedIn ':
                counts[("Observation","Step")] += 1

    # -------------------
    # 8. ToolCall -> Observation 
    # Pattern utile per verificare output dei tools 
    for toolcall in [n for n,d in G.nodes(data=True) if d['type']=='ToolCall']:
        for obs in G.successors(toolcall):
            if G.nodes[obs]['type']=='Observation' and G.edges[toolcall,obs]['predicate'] == 'obs:results':
                counts[("ToolCall","Observation")] += 1
    
    # -------------------
    # 9. Run -> TerminationSignal
    # Verifica che la Run, o il task venga effettivamente portato a terminazione
    for run in [n for n,d in G.nodes(data=True) if d['type']=='Run']:
        for termination in G.successors(run):
            if G.nodes[termination]['type']=='TerminationSignal' and G.edges[run,termination]['predicate']=='trace:endsWith':
                counts[("Observation","Step")] += 1
    
    # ======================
    # Motifs di lunghezza 3
    # ======================

    # -------------------
    # 10. Goal -> Intention -> ToolCall
    # Un Goal motiva un'Intention che si concretizza in una ToolCall
    for goal in [n for n,d in G.nodes(data=True) if d['type']=='Goal']:
        for intention in G.successors(goal):
            if G.nodes[intention]['type'] == 'Intention' and G.edges[goal,intention]['predicate'] == 'mind:motivates':
                for toolcall in G.successors(intention):
                    if G.nodes[toolcall]['type']=='ToolCall' and G.edges[intention,toolcall]['predicate'] == 'mind:commitsToToolCall':
                        counts[("Goal","Intention","ToolCall")] += 1

    # -------------------
    # 11. Intention -> ToolCall -> Observation
    # Un'Intention porta a una ToolCall che produce un'Observation
    for intention in [n for n,d in G.nodes(data=True) if d['type']=='Intention']:
        for toolcall in G.successors(intention):
            if G.nodes[toolcall]['type']=='ToolCall' and G.edges[intention,toolcall]['predicate'] == 'mind:commitsToToolCall':
                for obs in G.successors(toolcall):
                    if G.nodes[obs]['type']=='Observation' and G.edges[toolcall,obs]['predicate'] == 'obs:results':
                        counts[("Intention","ToolCall","Observation")] += 1


    # -------------------
    # 12. Run -> Step -> TerminationSignal
    # Verifica la sequenza di esecuzione completa: Run composta da Steps e termina con TerminationSignal
    for run in [n for n,d in G.nodes(data=True) if d['type']=='Run']:
        for step in G.successors(run):
            if G.nodes[step]['type']=='Step' and G.edges[run,step]['predicate']=='trace:hasStep':
                for term in G.successors(run):
                    if G.nodes[term]['type']=='TerminationSignal' and G.edges[run,term]['predicate']=='trace:endsWith':
                        counts[("Run","Step","TerminationSignal")] += 1
    
    # ======================
    # Motifs di lunghezza 4
    # ======================
    
    # -------------------
    # 13. Goal -> Intention -> ToolCall -> Observation
    # Catena completa: un Goal motiva un'Intention che porta a una ToolCall che produce un'Observation
    for goal in [n for n,d in G.nodes(data=True) if d['type']=='Goal']:
        for intention in G.successors(goal):
            if G.nodes[intention]['type']=='Intention' and G.edges[goal,intention]['predicate']=='mind:motivates':
                for toolcall in G.successors(intention):
                    if G.nodes[toolcall]['type']=='ToolCall' and G.edges[intention,toolcall]['predicate']=='mind:commitsToToolCall':
                        for obs in G.successors(toolcall):
                            if G.nodes[obs]['type']=='Observation' and G.edges[toolcall,obs]['predicate']=='obs:results':
                                counts[("Goal","Intention","ToolCall","Observation")] += 1

    return counts


# Funzione principale
def extract_subgraph_features(input_folder, output_file="motifs_counts.csv"):
    """
    Estrae i motifs da tutti i JSON presenti in input_folder
    e salva i conteggi in un CSV
    """
    all_data = []
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    
    for file_path in json_files:
        data = load_json(file_path)
        G = build_graph(data)
        counts = count_motifs(G) 
        
        # Prepara il dizionario da salvare, includendo la label (risultato del task)
        row = { "run_id": os.path.basename(file_path).replace(".json",""), "label": data.get("label","unknown") }
        for motif, c in counts.items():
            key = "->".join(motif)
            row[key] = c
        all_data.append(row)
    
    # Converte in DataFrame e salva CSV
    df = pd.DataFrame(all_data)
    df.fillna(0, inplace=True)  # motif assenti = 0
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / output_file
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved motif counts for {len(json_files)} runs to {output_file}")


# MAIN
if __name__ == "__main__":
    base_dir = Path(__file__).parent
    input_folder = base_dir / "json_graphs"  # cartella con i JSON dei KGs
    
    extract_subgraph_features(input_folder)