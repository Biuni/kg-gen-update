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

    # Access graph attributes safely; provide defaults if empty
    entities = graph.entities or {}
    edges = graph.edges or set()
    relations = graph.relations or set()
    
    # Print all entities
    print("\nEntities:")
    if entities:
        for key in sorted(entities.keys()):
            print(f"\n  - {key}:")
            value = entities[key]
            if isinstance(value, dict):
                print(json.dumps(value, indent=6))
            else:
                value_str = str(value).replace("\n", "\n      ")
                print(f"      {value_str}")
    else:
        print("  (none)")
    
    # Print all edges
    print("\nEdges:")
    if edges:
        for edge in sorted(edges):
            print(f"  - {edge}")
    else:
        print("  (none)")
    
    # Print all relations (source -- relation --> target)
    print("\nRelations:")
    if relations:
        for rel in sorted(relations):
            print(f"  - {rel[0]} -- {rel[1]} --> {rel[2]}")
    else:
        print("  (none)")

    # Print entity clusters, if any
    if graph.entity_clusters:
        print("\nEntity Clusters:")
        for cluster, members in graph.entity_clusters.items():
            print(f"  - {cluster}: {members}")
    
    # Print edge clusters, if any
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
    # Test: Graph generation + Aggregation using different reasoning traces
    ###########################################################################

    # SCENARIO 1: Simple calculator usage
    # ------------------------------------
    # Tests creation of a graph where a calculation is performed using an external
    # tool. Ensures entities, steps, and tool outputs are captured correctly.
    reasonings_1 = {
        1:"""        
        @agent(id="1", model="ollama")

        @state(
            id="2",
            current_step_id="1",
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
        
        @call(tool="Calculator")

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
        """
    }

    # SCENARIO 2: Search + Calculator
    # --------------------------------
    # Tests a scenario combining information retrieval and arithmetic calculation.
    # Checks if reasoning steps correctly call multiple tools and integrate results.
    reasonings_2 = {
    1:"""        
    @agent(id="10", model="ollama")

    @state(
        id="20",
        current_step_id="1",
        calculator_ready="true",                  
        search_ready="true",                     
        database_connected="false",               
        last_user_input="What is the current population of France and what is 5% of it?",  
        current_step_index="1",                   
        intentions_pending="['SearchPopulation']",  
        last_action_intent="SearchPopulation",      
        last_tool_call="None",         
        last_tool_output="None",                  
        termination_flag="false"                  
    )

    @user(id="30")
    @task(id="T100", description="Find the population of France and compute 5% of it")
    
    @run(id="50")

    @step(id="1", index="1")
    
    @start_reasoning(step_id="1")
    The user is asking for the current population of France.
    I do not have this memorized with guaranteed freshness.
    I should search for the latest population data.
    Using the search tool will allow me to retrieve up-to-date information.
    I will call the search tool with query "current population of France".
    @end_reasoning(step_id="1")

    @call(tool="search")

    @observation(
       tool="Search",
       output="Population of France in 2025 is approximately 68,000,000"
    )
    """,

    2:"""
    @step(id="2", index="2")

    @start_reasoning(step_id="2")
    I have received the population value: 68,000,000.
    Now I need to compute 5% of this number.
    5 percent means multiplying by 0.05.
    To ensure precision, I will use the calculator tool.
    I will call the calculator with the expression "68000000 * 0.05".
    @end_reasoning(step_id="2")
    
    @call(tool="calculator")

    @observation(
       tool="Calculator",
       output="3400000"
    )
    """,

    3:"""
    @step(id="3", index="3")

    @start_reasoning(step_id="3")
    The calculator returned 3,400,000.
    This represents 5% of the French population.
    I can now provide the final answer to the user.
    Answer provided.
    @end_reasoning(step_id="3")

    @termination
    """
    }

    # SCENARIO 3: Database + Calendar
    # --------------------------------
    # Tests database verification and scheduling an event.
    # Ensures that tool calls and their outputs are represented as entities/edges.
    reasonings_3 = {
    1:"""        
    @agent(id="11", model="ollama")

    @state(
        id="21",
        current_step_id="1",
        calculator_ready="false",                  
        search_ready="false",                     
        database_connected="true",               
        last_user_input="Schedule a meeting with John Doe tomorrow at 3pm if he exists in the database",  
        current_step_index="1",                   
        intentions_pending="['CheckUserInDatabase']",  
        last_action_intent="CheckUserInDatabase",      
        last_tool_call="None",         
        last_tool_output="None",                  
        termination_flag="false"                  
    )

    @user(id="31")
    @task(id="T200", description="Schedule meeting with John Doe tomorrow at 3pm if user exists")
    
    @run(id="51")

    @step(id="1", index="1")
    
    @start_reasoning(step_id="1")
    The task requires verifying whether John Doe exists in the database.
    The database connection is available.
    I should query the database for a user named John Doe.
    I will call the database tool with query "SELECT * FROM users WHERE name='John Doe'".
    @end_reasoning(step_id="1")

    @call(tool="Database")

    @observation(
       tool="Database",
       output="User found: id=789, name=John Doe"
    )
    """,

    2:"""
    @step(id="2", index="2")

    @start_reasoning(step_id="2")
    The database confirms that John Doe exists.
    The next goal is to schedule the meeting.
    I intend to create a calendar event for tomorrow at 3pm.
    I will call the calendar tool with title "Meeting with John Doe" and time "2026-02-13 15:00".
    @end_reasoning(step_id="2")

    @call(tool="Calendar)

    @observation(
       tool="Calendar",
       output="Event created successfully with id=EVT123"
    )
    """,

    3:"""
    @step(id="3", index="3")

    @start_reasoning(step_id="3")
    The calendar event was created successfully.
    The task is now complete.
    No further actions needed.
    @end_reasoning(step_id="3")

    @termination
    """
    }

    # SCENARIO 4: Shell + Search
    # ---------------------------
    # Tests system command execution and conditional external search.
    # Focuses on reasoning steps that depend on tool outputs to decide next actions.
    reasonings_4 = {
    1:"""        
    @agent(id="12", model="ollama")

    @state(
        id="22",
        current_step_id="1",
        calculator_ready="false",                  
        search_ready="true",                     
        database_connected="false",               
        last_user_input="Check disk usage and tell me if more than 80% is used. If so, search how to clean temp files.",  
        current_step_index="1",                   
        intentions_pending="['CheckDiskUsage']",  
        last_action_intent="CheckDiskUsage",      
        last_tool_call="None",         
        last_tool_output="None",                  
        termination_flag="false"                  
    )

    @user(id="32")
    @task(id="T300", description="Check disk usage and search cleaning instructions if above threshold")
    
    @run(id="52")

    @step(id="1", index="1")
    
    @start_reasoning(step_id="1")
    The user wants to check disk usage.
    This requires executing a system command.
    I will use the shell tool to run "df -h".
    @end_reasoning(step_id="1")
    
    @call(tool="Shell")

    @observation(
       tool="Shell",
       output="Disk usage: 85%"
    )
    """,

    2:"""
    @step(id="2", index="2")

    @start_reasoning(step_id="2")
    The disk usage is 85 percent.
    This exceeds the 80 percent threshold specified by the user.
    Therefore, I need to search for instructions to clean temporary files.
    I will call the search tool with query "how to clean temporary files linux".
    @end_reasoning(step_id="2")

    @call(tool="Search")

    @observation(
       tool="Search",
       output="You can remove temporary files using rm -rf /tmp/*"
    )
    """,

    3:"""
    @step(id="3", index="3")

    @start_reasoning(step_id="3")
    I have retrieved instructions for cleaning temporary files.
    I will present these instructions to the user.
    Task complete.
    @end_reasoning(step_id="3")

    @termination
    """
    }
    
    # SCENARIO 5: Single tool, discursive reasoning
    # ---------------------------------------------
    # Tests a long-form reasoning scenario with only the shell tool.
    # Highlights chronological ordering of steps, intentions, and reasoning statements.
    reasonings_5 = {

    1: """
    @agent(id="35", model="ollama")

    @state(
        id="45",
        current_step_id="1",
        calculator_ready="false",
        search_ready="false",
        database_connected="false",
        last_user_input="Check if disk usage exceeds 80% and explain whether cleanup is needed.",
        current_step_index="1",
        intentions_pending="['EvaluateDiskUsage']",
        last_action_intent="EvaluateDiskUsage",
        last_tool_call="None",
        last_tool_output="None",
        termination_flag="false"
    )

    @user(id="55")
    @task(id="T550", description="Evaluate disk usage and determine if cleanup is required")

    @run(id="75")

    @step(id="1", index="1")

    @start_reasoning(step_id="1")
    The user is requesting an evaluation of the current disk usage.
    The request is conditional because it depends on whether the usage exceeds eighty percent.
    Therefore, the first requirement is to obtain the current disk usage value.
    There is no stored disk usage information in the current state.
    This means that relying on memory would be unreliable.
    The only dependable way to retrieve accurate and up-to-date information
    is to query the operating system directly.
    The appropriate tool for interacting with the operating system
    is the shell tool.
    The shell tool allows execution of system-level commands.
    Disk usage is system-level information.
    Therefore, the shell tool is the correct and sufficient instrument.
    There is no need for external search.
    There is no need for database access.
    There is no need for calculation at this stage.
    Only direct system inspection is required.
    Before making any judgment about cleanup necessity,
    I must gather the numeric value.
    Once I have that value,
    I will compare it against the eighty percent threshold.
    I will now call the shell tool to retrieve the disk usage statistic.
    @end_reasoning(step_id="1")

    @call(tool="Shell")

    @observation(
    tool="Shell",
    output="Disk usage: 84%"
    )
    """,

    2: """
    @step(id="2", index="2")

    @start_reasoning(step_id="2")
    The shell tool returned a disk usage value of eighty-four percent.
    The threshold specified by the user is eighty percent.
    Eighty-four percent is greater than eighty percent.
    This means the condition described by the user is satisfied.
    Therefore, disk usage currently exceeds the acceptable threshold.
    Now I must determine whether cleanup is needed.
    If disk usage exceeds the threshold,
    cleanup is not only recommended but advisable.
    High disk usage can lead to performance degradation.
    High disk usage can also increase the risk of system instability.
    Additionally, limited free space may prevent new files from being written.
    If the disk usage had been below eighty percent,
    cleanup would not have been urgent.
    However, that is not the case.
    The observed value clearly exceeds the defined boundary.
    Therefore, cleanup actions should be recommended.
    These actions may include removing unnecessary temporary files,
    clearing old logs,
    or deleting unused applications.
    The evaluation has been completed.
    The threshold comparison has been performed.
    A justified conclusion has been reached.
    No further tool interaction is necessary.
    The objective of the task has been fully satisfied.
    @end_reasoning(step_id="2")

    @termination
    """
    }

    # SCENARIO 6: Multi-step temperature evaluation
    # ---------------------------------------------
    # Tests combination of search and calculator tools for a conditional numerical task.
    reasonings_6 = {

    1: """
    @agent(id="88", model="ollama")

    @state(
        id="200",
        current_step_id="1",
        calculator_ready="true",
        search_ready="true",
        database_connected="false",
        last_user_input="Evaluate temperature and possibly compute missing degrees",
        current_step_index="1",
        intentions_pending="['EvaluateDiskUsage']",
        last_action_intent="EvaluateDiskUsage",
        last_tool_call="None",
        last_tool_output="None",
        termination_flag="false"
    )

    @user(id="300")
    @task(id="T900", description="Evaluate temperature and possibly compute missing degrees")

    @run(id="400")

    @step(id="1", index="1")

    @start_reasoning(step_id="1")
    The user wants to know the current temperature in Rome.
    To obtain real-time temperature data, I must use the search tool.
    The search tool is appropriate for retrieving up-to-date weather information.
    No calculation is required at this stage.
    I will call the search tool to retrieve today's temperature in Rome.
    @end_reasoning(step_id="1")

    @call(tool="Search")

    @observation(
    tool="Search",
    output="Current temperature in Rome: 7°C"
    )
    """,

    2: """
    @step(id="2", index="2")

    @start_reasoning(step_id="2")
    The retrieved temperature is seven degrees Celsius.
    The condition specified by the user is below ten degrees.
    Seven is less than ten.
    Therefore, the condition is satisfied.
    Since the temperature is below ten degrees,
    I must calculate how many degrees are missing to reach twenty degrees.
    This requires subtraction.
    The calculator tool is appropriate for arithmetic operations.
    I will compute twenty minus seven.
    @end_reasoning(step_id="2")

    @call(tool="Calculator")

    @observation(
    tool="Calculator",
    output="20 - 7 = 13"
    )
    """,

    3: """
    @step(id="3", index="3")

    @start_reasoning(step_id="3")
    The calculator returned thirteen.
    This means thirteen degrees are missing to reach twenty degrees.
    The objective of the task has been fulfilled.
    No further tool calls are necessary.
    I will now terminate the run.
    @end_reasoning(step_id="3")

    @termination
    """
    }
    
    # SCENARIO 7: Single-step arithmetic reasoning
    # ---------------------------------------------
    # Simple reasoning without tools. Ensures minimal reasoning steps are captured correctly.
    reasonings_7 = {
    1: """
    @agent(id="88", model="ollama")

    @state(
        id="200",
        current_step_id="1",
        calculator_ready="true",
        search_ready="true",
        database_connected="false",
        last_user_input="Evaluate a simple arithmetic reasoning",
        current_step_index="1",
        intentions_pending="['SimpleArithmetic']",
        last_action_intent="SimpleArithmetic",
        last_tool_call="None",
        last_tool_output="None",
        termination_flag="false"
    )

    @user(id="300")
    @task(id="T1000", description="Perform a single-step reasoning without external tools")

    @run(id="400")

    @step(id="1", index="1")

    @start_reasoning(step_id="1")
    The user wants to know if 7 is less than 10 and how much is missing to reach 20.
    Seven is indeed less than ten, so the first condition is satisfied.
    To reach twenty, thirteen more is needed.
    All of this can be reasoned without using any external tool.
    The task objective has been fulfilled.
    No further steps or tool calls are necessary.
    I will now terminate the run.
    @end_reasoning(step_id="1")

    @termination
    """
    }

    # SCENARIO 8: Multi-Goal, Multi-Tool reasoning
    # ---------------------------------------------
    # Complex scenario where multiple goals are evaluated and multiple tools are used.
    # Tests if goals, steps, and intermediate calculations are captured correctly.
    reasonings_8 = {
        1: """
        @agent(id="101", model="ollama")

        @state(
            id="301",
            current_step_id="1",
            calculator_ready="true",
            search_ready="true",
            database_connected="true",
            last_user_input="Check disk usage and temperature, and suggest actions accordingly",
            current_step_index="1",
            intentions_pending="['EvaluateDiskUsage','EvaluateTemperature']",
            last_action_intent="EvaluateDiskUsage",
            last_tool_call="None",
            last_tool_output="None",
            termination_flag="false"
        )

        @user(id="401")
        @task(id="T800", description="Evaluate disk usage and temperature, recommend actions")

        @run(id="500")

        @step(id="1", index="1")
        @start_reasoning(step_id="1")
        The user wants to check if the system requires maintenance based on disk usage and temperature. First of all,
        I must ensure disk usage is below 80%. Then, the second goal is to ensure temperature is within safe limits (<75°C).
        I will first call the shell tool to retrieve disk usage.
        @end_reasoning(step_id="1")

        @call(tool="Shell")

        @observation(
            tool="Shell",
            output="Disk usage: 82%"
        )
        """,
        2:"""
        @step(id="2", index="2")
        @start_reasoning(step_id="2")
        Disk usage exceeds 80%, so the first Goal is not satisfied.
        I need to suggest cleanup actions.
        Next, I will check the current system temperature using the search tool.
        @end_reasoning(step_id="2")

        @call(tool="Search")

        @observation(
            tool="Search",
            output="Current CPU temperature: 70°C"
        )
        """,
        3:"""
        @step(id="3", index="3")
        @start_reasoning(step_id="3")
        Temperature is 70°C, which is within safe limits (<75°C), so the second Goal is satisfied.
        Since disk usage exceeds threshold, I will calculate estimated free space after cleanup using the calculator tool.
        @end_reasoning(step_id="3")

        @call(tool="Calculator")

        @observation(
            tool="Calculator",
            output="82% - 20% = 62%"
        )
        """,
        4:"""
        @step(id="4", index="4")
        @start_reasoning(step_id="4")
        After cleanup, disk usage would reduce to 62%, achieving the first Goal.
        Both Goals are now satisfied.
        Recommended actions:
        - Run cleanup scripts to free disk space.
        - No action needed for temperature.
        Task complete. All Goals achieved.
        @end_reasoning(step_id="4")

        @termination
        """
    }

    # SCENARIO 9: Multi-goal discursive reasoning
    reasonings_9 = {
    1: """
    @agent(id="101", model="ollama")

    @state(
        id="500",
        current_step_id="1",
        calculator_ready="true",
        search_ready="true",
        database_connected="true",
        last_user_input="Check the system status and provide recommendations if something is off",
        current_step_index="1",
        intentions_pending="['CheckDiskAndCPU']",
        last_action_intent="CheckDiskAndCPU",
        last_tool_call="None",
        last_tool_output="None",
        termination_flag="false"
    )

    @user(id="401")
    @task(id="T900", description="Check system metrics and suggest actions")

    @run(id="600")

    @step(id="1", index="1")

    @start_reasoning(step_id="1")
    "Before proceeding, I want to make sure the system is healthy. My first goal is to check that disk usage is below 80%, to avoid 
    slowdowns or performance issues. The second goal is to ensure the CPU temperature stays under 75°C, preventing overheating. 
    To achieve these goals, I will use the Shell tool to check disk usage and a search tool to measure the CPU temperature. 
    If the disk is nearly full, I will suggest cleanup operations; if the CPU is too hot, I will recommend cooling measures. 
    By planning both checks together, I can ensure the system is operating safely and suggest concrete actions if needed."
    @end_reasoning(step_id="1")

    @call(tool="Shell")
    @observation(
        tool="Shell",
        output="Disk usage: 85%"
    )

    @call(tool="Search")
    @observation(
        tool="Search",
        output="CPU temperature: 78°C"
    )

    @termination
    """
    }
    
    # Dictionary of all reasoning scenarios
    reasonings_dict = {
    1: reasonings_1,  # Simple calculator
    2: reasonings_2,  # Search + Calculator
    3: reasonings_3,  # Database + Calendar
    4: reasonings_4,  # Shell + Search
    5: reasonings_5,  # Shell
    6: reasonings_6,  # Multi-step temperature eval
    7: reasonings_7,  # Single-step arithmetic
    8: reasonings_8,  # Multi-Goal, Multi-Tool
    9: reasonings_9,  # Multi-Goal 
    }
    
    # Prompt user to select a scenario to test
    print("Available reasoning scenarios:")
    for k in sorted(reasonings_dict.keys()):
        print(f"  {k}: Scenario {k}")

    choice = input("Enter the scenario number you want to test: ").strip()
    if choice.isdigit() and int(choice) in reasonings_dict:
        selected_reasonings = reasonings_dict[int(choice)]
    else:
        print("Invalid choice. Exiting.")
        exit()

    # Generate Knowledge Graphs for each reasoning step
    graphs = {}

    for i, reasoning in selected_reasonings.items():
        print(f"\n=== Generating reasoning step {i} ===")
        start = time.time()
        graph = kg.generate(input_data=reasoning, context=context)
        end = time.time()
        print(f"Graph generation time (step {i}): {end - start:.2f}s")
        graphs[i] = graph
        pretty_print_graph(graph, f"GRAPH Step {i}")
        kg.visualize(graph, f"graph_step_{i}.html", open_in_browser=True)
 

    # Aggregate all graphs for the selected scenario
    # ----------------------------------------------
    # This function combines all individual step graphs into a single, unified knowledge graph.
    # Even if the scenario has only one reasoning step, aggregation is important because:
    # 1. It ensures that edges connecting the 'run' node to each step are added.
    # 2. It ensures that the 'termination' node is linked to the last step.
    # 3. Without aggregation, these connecting edges would be missing, and the graph
    #    would not correctly represent the workflow of the reasoning process.
    aggregated_graph = kg.aggregate(list(graphs.values()))
    pretty_print_graph(aggregated_graph, "Aggregated Graph")
    kg.visualize(aggregated_graph, "aggregated_graph.html", open_in_browser=True)


       
    



