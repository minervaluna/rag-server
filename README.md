# Intro

1. 实现一个 Agentic RAG 项目
2. 使用 LangGraph LangChain LlamaIndex

# Archive

## workflow

```
graph TD
    A[用户输入] --> B[LangGraph 控制流]
    B --> C[LangChain Agent 决策]
    C --> D[LlamaIndex 检索文档]
    D --> E[LangChain LLM 回答生成]
```

## general archive

```
User Query
   ↓
LangGraph Agent Executor
   ↓
[ Tools / Nodes ]
   ↙          ↓            ↘
RAG Retriever   Memory      LLM Reasoning
   ↓                            ↓
LlamaIndex Query      →     LangChain Agent
   ↓                            ↓
  Documents          ←      Final Response
```

# components

| 组件               | 作用说明                                       |
|------------------|--------------------------------------------|
| **LangGraph**    | 构建 Agentic 工作流，定义 node、edge、state 的执行逻辑    |
| **LangChain**    | 用于管理 Agent、Tool、Memory 的行为逻辑               |
| **LlamaIndex**   | 实现 RAG 检索能力，作为自定义 Tool 接入 Agent            |
| **LLM Provider** | 智谱 AI（GLM 系列），通过 LangChain 或自定义 wrapper 接入 |

# Refers

* [Agentic RAG - from LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
* [Agentic RAG：概念、类型、应用与实现](https://zhuanlan.zhihu.com/p/17693009207)
