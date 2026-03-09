"""
Missing Pattern Detection
----------------------------
Rileva anomalie cognitive nei Knowledge Graph delle run
identificando pattern attesi mancanti.

Missing Pattern Detection:
-> Run without ToolCall
-> Run without Termination
-> Run without ToolCall or Termination
-> Goal without Intention
-> Intention without ToolCall
-> ToolCall without Observation
-> ToolCall with Null Observation
-> Step without Intention
-> Step without Goal
-> Run without Goal -> Intention -> ToolCall
-> Run without Intention -> ToolCall -> Observation
-> Run without Goal -> Intention -> ToolCall -> Observation
"""

import networkx as nx
import json
from collections import defaultdict
import os
from pathlib import Path
import glob
import pandas as pd


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# Build Graph
def build_graph(data):
    """
    Costruisce un grafo NetworkX dal KG JSON.
    """
    G = nx.DiGraph()

    for n in data["nodes"]:
        node_type = n.split(":")[1].split("_")[0]
        G.add_node(n, type=node_type)

    for e in data["edges"]:
        G.add_edge(
            e["source"],
            e["target"],
            predicate=e.get("predicate")
        )

    return G


# Missing Pattern Detection
def detect_missing_patterns(G):
    """
    Cerca violazioni delle invarianti cognitive dell'agente.
    """

    anomalies = defaultdict(int)

    # Run Without ToolCall
    toolcalls = [
        n for n, d in G.nodes(data=True)
        if d["type"] == "ToolCall"
    ]

    if len(toolcalls) == 0:
        anomalies["run_without_toolcall"] += 1

    # Run Without Termination
    termination = [
        n for n, d in G.nodes(data=True)
        if d["type"] == "TerminationSignal"
    ]

    if len(termination) == 0:
        anomalies["run_without_termination"] += 1

    # Run Without ToolCall OR Termination
    if (len(toolcalls) == 0 or len(termination) == 0):
        anomalies["run_without_toolcall_or_termination"] += 1

    # Goal Without Intention
    for goal in [
        n for n, d in G.nodes(data=True)
        if d["type"] == "Goal"
    ]:

        has_intention = any(
            G.nodes[s]["type"] == "Intention"
            and G.edges[goal, s].get("predicate") == "mind:motivates"
            for s in G.successors(goal)
        )

        if not has_intention:
            anomalies["goal_without_intention"] += 1

    # Intention Without ToolCall
    for intention in [
        n for n, d in G.nodes(data=True)
        if d["type"] == "Intention"
    ]:

        has_toolcall = any(
            G.nodes[s]["type"] == "ToolCall"
            and G.edges[intention, s].get("predicate")
            == "mind:commitsToToolCall"
            for s in G.successors(intention)
        )

        if not has_toolcall:
            anomalies["intention_without_toolcall"] += 1

    # ToolCall without Observation
    for tc in [
        n for n, d in G.nodes(data=True)
        if d["type"] == "ToolCall"
    ]:

        observations = [
            s for s in G.successors(tc)
            if G.nodes[s]["type"] == "Observation"
            and G.edges[tc, s].get("predicate") == "obs:results"
        ]

        if len(observations) == 0:
            anomalies["toolcall_without_observation"] += 1

    # ToolCall with Null Observation (da rivedere)
    for n, d in G.nodes(data=True):
        if d["type"] == "Observation" and "_None_" in n:
            anomalies["null_observation_present"] += 1

    # Step without Intention
    for step in [
        n for n, d in G.nodes(data=True)
        if d["type"] == "Step"
    ]:
        intentions = [
            s for s in G.successors(step)
            if G.nodes[s]["type"] == "Intention"
            and G.edges[step,s].get("predicate") == "mind:expressedIn"
        ]

        if len(intentions) == 0:
            anomalies["step_without_intention"] += 1

    # Step without Goal
    for step in [
        n for n, d in G.nodes(data=True)
        if d["type"] == "Step"
    ]:
        goals = [
            s for s in G.successors(step)
            if G.nodes[s]["type"] == "Goal"
            and G.edges[step,s].get("predicate") == "mind:goalInStep"
        ]

        if len(goals) == 0:
            anomalies["step_without_goal"] += 1

    # Run without Goal -> Intention -> ToolCall
    goals = [n for n, d in G.nodes(data=True) if d["type"] == "Goal"]
    found_goal_int_toolcall = False
    for goal in goals:
        for intention in G.successors(goal):
            if G.nodes[intention]["type"] == "Intention" and G.edges[goal, intention].get("predicate") == "mind:motivates":
                for tc in G.successors(intention):
                    if G.nodes[tc]["type"] == "ToolCall" and G.edges[intention, tc].get("predicate") == "mind:commitsToToolCall":
                        found_goal_int_toolcall = True
                        break
    if not found_goal_int_toolcall:
        anomalies["missing_goal_intention_toolcall"] += 1

    # Run without Intention -> ToolCall -> Observation
    intentions = [n for n, d in G.nodes(data=True) if d["type"] == "Intention"]
    found_int_toolcall_obs = False
    for intention in intentions:
        for tc in G.successors(intention):
            if G.nodes[tc]["type"] == "ToolCall" and G.edges[intention, tc].get("predicate") == "mind:commitsToToolCall":
                for obs in G.successors(tc):
                    if G.nodes[obs]["type"] == "Observation" and G.edges[tc, obs].get("predicate") == "obs:results":
                        found_int_toolcall_obs = True
                        break
    if not found_int_toolcall_obs:
        anomalies["missing_intention_toolcall_observation"] += 1

    # Run without Goal -> Intention -> ToolCall -> Observation
    found_full_chain = False
    for goal in goals:
        for intention in G.successors(goal):
            if G.nodes[intention]["type"] == "Intention" and G.edges[goal, intention].get("predicate") == "mind:motivates":
                for tc in G.successors(intention):
                    if G.nodes[tc]["type"] == "ToolCall" and G.edges[intention, tc].get("predicate") == "mind:commitsToToolCall":
                        for obs in G.successors(tc):
                            if G.nodes[obs]["type"] == "Observation" and G.edges[tc, obs].get("predicate") == "obs:results":
                                found_full_chain = True
                                break
    if not found_full_chain:
        anomalies["missing_goal_intention_toolcall_observation"] += 1


    return anomalies


def extract_missing_patterns(input_folder, output_file="missing_patterns.csv"):
    all_rows = []

    json_files = glob.glob(os.path.join(input_folder, "*.json"))

    for file_path in json_files:
        data = load_json(file_path)
        G = build_graph(data)

        anomalies = detect_missing_patterns(G)

        row = {
            "run_id": os.path.basename(file_path).replace(".json", ""),
            "label": data.get("label", "unknown")
        }

        row.update(anomalies)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.fillna(0, inplace=True)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / output_file
    df.to_csv(output_path, index=False)

    print(f"[INFO] Missing pattern analysis saved to {output_path}")


if __name__ == "__main__":

    base_dir = Path(__file__).parent
    input_folder = base_dir / "json_graphs"

    extract_missing_patterns(input_folder)