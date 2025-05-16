from langchain.tools import Tool

def get_rag_tool(index):
    def rag_search(query: str) -> str:
        return index.query(query).response

    return Tool(
        name="RAG Search",
        func=rag_search,
        description="Searches documents using retrieval-augmented generation."
    )