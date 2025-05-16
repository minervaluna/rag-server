from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

def build_agent(llm, tools):
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )