import os
import time

from dotenv import load_dotenv
from kg_gen import KGGen

load_dotenv()

if __name__ == "__main__":

    kg = KGGen(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        api_base=os.getenv("API_BASE"),
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )

    context = "LLM Reasoning"
    reasoning1 = """
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
    """
    reasoning2 = """
    Okay, let me see. The user asked to multiply 20 by 30.1 and then divide the result by 10 plus 3.
    I need to make sure I did the calculation correctly. First, 20 multiplied by 30.1. Let me do that step again.
    20 times 30 is 600, and 20 times 0.1 is 2, so adding those gives 602. That seems right.
    Next, the denominator is 10 plus 3, which is 13. So then, 602 divided by 13.
    Let me check that division. 13 times 46 is 598, and 602 minus 598 is 4.
    So 4/13 is approximately 0.3077. Adding that to 46 gives 46.3077, which matches the tool's result of 46.30769230769231. 
    The user might be working on a math problem or needing a quick calculation.
    They might not have a calculator handy, so providing the precise result is important.
    Also, they might appreciate a brief explanation to confirm the steps.
    I should present the answer clearly and offer further help in case they have more questions.
    """

    start = time.time()
    graph1 = kg.generate(
        input_data=reasoning1,
        context=context,
    )
    end = time.time()
    print('Generazione reasoning 1: ', end - start)

    start = time.time()
    graph2 = kg.generate(
        input_data=reasoning2,
        context=context,
    )
    end = time.time()
    print('Generazione reasoning 2: ', end - start)

    start = time.time()
    aggregated_graph = kg.aggregate([graph1, graph2])
    end = time.time()
    print('Aggregazione dei grafi: ', end - start)

    start = time.time()
    clustered_graph1 = kg.deduplicate(
        aggregated_graph,
        context=context,
        method="full"
    )
    end = time.time()
    print('Deduplicazione LLM Based su grafo aggregato: ', end - start)

    start = time.time()
    clustered_graph2 = kg.deduplicate(
        aggregated_graph,
        context=context,
        method="semhash"
    )
    end = time.time()
    print('Deduplicazione SEMHASH su grafo aggregato: ', end - start)

    print(graph1)
    print(graph2)
    print(aggregated_graph)
    print(clustered_graph1)
    print(clustered_graph2)

    kg.visualize(graph1, "graph1.html")
    kg.visualize(graph2, "graph2.html")
    kg.visualize(aggregated_graph, "aggregated_graph.html")
    kg.visualize(clustered_graph1, "clustered_graph1.html", open_in_browser=True)
    kg.visualize(clustered_graph2, "clustered_graph2.html", open_in_browser=True)