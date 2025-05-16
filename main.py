from dotenv import load_dotenv

from agents.base_agent import build_agent
from agents.tools import get_rag_tool
from graph.rag_graph import build_rag_graph
from models.llm_loader import load_llm
from retriever.index import build_index


def main():
    load_dotenv()
    index = build_index()
    rag_tool = get_rag_tool(index)
    llm = load_llm()
    agent_executor = build_agent(llm=llm, tools=[rag_tool])
    rag_graph = build_rag_graph(agent_executor)

    print("Agentic RAG ready. Type 'exit' to quit.")
    while True:
        query = input("\nUser: ")
        if query.lower() in {"exit", "quit"}:
            break
        result = rag_graph.invoke({"input": query})
        print("\nAgent:", result.get("result"))


if __name__ == "__main__":
    main()
