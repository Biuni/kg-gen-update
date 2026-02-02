from dotenv import load_dotenv
from src.kg_gen import KGGen
import os

load_dotenv()

def kg():
    return KGGen(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE")),
        api_base=os.getenv("API_BASE"),
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )


def test_custom_api_base(kg: KGGen):
    text = """
    Harry has two parents - his dad James Potter and his mom Lily Potter.
    Harry and his wife Ginny have three kids together: 
    their oldest son James Sirius, their other son Albus, and their daughter Lily Luna.
    """
    graph = kg.generate(
        input_data=text
    )
    print(graph)

def test_aggregation(kg: KGGen, context: str, text1: str, text2: str):
    graph1 = kg.generate(
        input_data=text1,
        context=context,
    )
    print("Graph_1:")
    print(graph1)

    graph2 = kg.generate(
        input_data=text2,
        context=context,
    )
    print("Graph_2:")
    print(graph1)

    combined_graph = kg.aggregate([graph1, graph2])
    print("Combined:")
    print(combined_graph)

    clustered_graph = kg.deduplicate(
        combined_graph,
        context=context,
    )
    print("Clustered:")
    print(clustered_graph)

if __name__ == "__main__":
    #test_custom_api_base(kg())
    context = """
    Knowledge Graph
    """
    text1 = """
    In knowledge representation and reasoning, a knowledge graph is a knowledge base that uses a graph-structured 
    data model or topology to represent and operate on data. Knowledge graphs are often used to store interlinked 
    descriptions of entities - objects, events, situations or abstract concepts - while also encoding the free-form 
    semantics or relationships underlying these entities.
    """
    text2 = """
    Since the development of the Semantic Web, knowledge graphs have often been associated with linked open data projects, 
    focusing on the connections between concepts and entities. They are also historically associated with and used 
    by search engines such as Google, Bing, and Yahoo; knowledge engines and question-answering services such as WolframAlpha, 
    Apple's Siri, and Amazon Alexa; and social networks such as LinkedIn and Facebook.
    """
    test_aggregation(
        kg(),
        context,
        text1,
        text2
    )
    
