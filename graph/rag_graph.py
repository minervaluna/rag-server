from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

def build_rag_graph(agent_executor):
    def run_agent(state):
        response = agent_executor.invoke(state["input"])
        return {"result": response["output"]}

    builder = StateGraph(dict)
    builder.add_node("agent", RunnableLambda(run_agent))
    builder.set_entry_point("agent")
    builder.add_edge("agent", END)
    return builder.compile()