import os
import time
import json
from dotenv import load_dotenv
from kg_gen import KGGen


# Load environment variables from .env file
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
    
    # Accesso diretto agli attributi Pydantic
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
                # gestisce multilinea con indentazione
                value_str = str(value).replace("\n", "\n      ")
                print(f"      {value_str}")
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

    # Eventuali cluster
    if graph.entity_clusters:
        print("\nEntity Clusters:")
        for cluster, members in graph.entity_clusters.items():
            print(f"  - {cluster}: {members}")

    if graph.edge_clusters:
        print("\nEdge Clusters:")
        for cluster, members in graph.edge_clusters.items():
            print(f"  - {cluster}: {members}")

    print("="*60 + "\n")



if __name__ == "__main__":

    kg = KGGen(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        api_base=os.getenv("API_BASE"),
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )

    context = "LLM Reasoning"
    
    ###########################################################################
    # Test: Graph generation + deduplication using different reasoning traces
    ###########################################################################

    # Example reasoning traces (LLM internal reasoning logs)
    reasonings = {
        1:"""        
        @agent(id="1", model="ollama")

        @state(
            id="2"
            calculator_ready="true",                  
            search_ready="false",                     
            database_connected="true",               
            last_user_input="Calculate 20 * 30.1 / 13",  
            current_step_index="1",                   
            intentions_pending="['CallCalculator']",  
            last_action_intent="CallCalculator",      
            last_tool_call="Tool_Calculator",         
            last_tool_output="None",                  
            termination_flag="false"                  
        )

        @user(id="3")
        @task(id="T1", description="Calculate the result of 20 * 30.1 / 13")
        
        @run(id="5")

        @step(id="1", index="50")
        
        @start_reasoning(step_id="1")
        Okay, let's see. The user wants to multiply 20 by 30.1 and then divide that result by (10 plus 3).
        First, I need to make sure I understand the order of operations here.
        The multiplication comes first, right? So 20 times 30.1. Let me calculate that.
        20 * 30 is 600, and 20 * 0.1 is 2, so adding those together gives 602.
        Then, the denominator is 10 + 3, which is 13. So now I need to divide 602 by 13. Let me do that division.
        13 times 46 is 598, so 602 minus 598 is 4.
        That leaves a remainder of 4, so the result is 46 with a remainder of 4, or as a decimal, approximately 46.3077.
        Wait, but the user might want the exact decimal or a fraction.
        Alternatively, maybe I should use the calculator function provided to get the precise result.
        Let me check the expression again: (20 * 30.1) / (10 + 3).
        Using the calc tool would handle the exact calculation, ensuring there's no error in my manual computation.
        I should call the function with the expression as a string.
        @end_reasoning(step_id="1")

        @observation(
           tool="Calculator",
           output="46.3076923077"
        )

        """,
        2:"""
        @trace.step(id="2", index="51")

        @start_reasoning(step_id="2")
        I have now received the output from the calculator tool: 46.3076923077.
        This matches my manual estimation, confirming the calculation is correct.
        The result can be presented to the user as the final answer.
        I will also make a note that the calculation is precise and verified.
        @end_reasoning(step_id="2")

        @termination
        """,
        # 3:"""
        # Okay, the user wants to multiply 20 by 30.1, then divide the result by 10 plus 3. Let's think carefully.
        # First, 20 times 30.1. I’ll break it down: 20 times 30 is 600, 20 times 0.1 is 2, so altogether that gives 602. 
        # That seems correct. Next, the denominator is 10 plus 3, which is 13. So 602 divided by 13. Doing the division, 13 goes
        # into 602 forty-six times with a remainder of 4. As a decimal, 46.3077 approximately. 
        # To be fully precise, I’ll use the calculator tool. Inputting "(20*30.1)/(10+3)" confirms 46.30769230769231. Perfect.
        # I can now provide this result to the user. Done with the computation.
        # """,
        # 4:"""
        # Let’s see, the user asked to multiply 20 by 30.1 and divide by 10 plus 3. I need to make sure I get the correct order
        # of operations. Multiplication first: 20 times 30.1. 20 times 30 is 600, 20 times 0.1 is 2, so 602 in total. Good.
        # Now divide by 10 + 3, which is 13. 602 divided by 13 gives 46 remainder 4, so approximately 46.3077. I could stop
        # here, but to ensure exact precision, I’ll use the calculator tool.
        # Feeding the expression into the calculator confirms 46.30769230769231. The user might want the exact 
        # decimal, so it’s good to double-check. I can now confidently present this result.
        # """,
    }

    graphs = {}

    # Generate graphs and apply deduplication strategies
    for i, reasoning in reasonings.items():
        print(f"\n=== Generating reasoning {i} ===")

        # Graph generation
        start = time.time()
        graph = kg.generate(input_data=reasoning, context=context)
        end = time.time()
        print(f"Graph generation time (reasoning {i}): {end - start:.2f}s")

        graphs[i] = graph
        
        # Print and visualize original graph
        pretty_print_graph(graph, f"GRAPH{i} - Reasoning {i}") 
        kg.visualize(graph, f"graph{i}.html", open_in_browser=True) 

    # Aggregation 
    if len(graphs) > 1:
        aggregated_graph = kg.aggregate(list(graphs.values()))
        pretty_print_graph(aggregated_graph, "Aggregated Graph")
    kg.visualize(aggregated_graph, "aggregated_graph.html", open_in_browser=True)

       