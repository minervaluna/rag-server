import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import ChatGLM, HuggingFaceHub, QianfanLLMEndpoint


def load_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    if provider == "openai":
        print(f"Loading OpenAI model: {model_name}")
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            openai_api_base=base_url or None
        )
    elif provider == "huggingface":
        print(f"Loading HuggingFace model: {model_name}")
        return HuggingFaceHub(
            huggingfacehub_api_token=api_key,
            repo_id=model_name,
            model_kwargs={"temperature": 0.7}
        )
    elif provider == "zhipuai":
        print(f"Loading ZhipuAI model: {model_name}")
        return ChatGLM(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=0.7
        )
    elif provider == "qwen":
        # 千问（阿里达摩院）
        return QianfanLLMEndpoint(
            qianfan_ak=os.getenv("QIANFAN_AK"),
            qianfan_sk=os.getenv("QIANFAN_SK"),
            model=model_name
        )

    elif provider == "deepseek":
        # DeepSeek 也是 OpenAI 协议
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            temperature=0.7,
            openai_api_base=base_url or None
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
