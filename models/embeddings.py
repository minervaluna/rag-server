from typing import Optional, Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field
from sentence_transformers import SentenceTransformer


class BGEEmbedding(BaseEmbedding):
    instruction: Optional[str] = Field(default="为这个句子生成表示以用于检索相关文章：")

    def __init__(self, /, model_name: str = "BAAI/bge-base-zh-v1.5", device: Optional[str] = None,
                 cache_dir: Optional[str] = None, instruction: Optional[str] = None,
                 **data: Any):
        """
        :param model_name: HuggingFace 模型名称，如 'BAAI/bge-m3' | 'BAAI/bge-base-zh-v1.5'
        :param device: 如 'cuda' or 'cpu'
        :param cache_dir: 缓存目录，默认用 huggingface 的 ~/.cache 路径
        :param instruction: 查询时的前缀提示，适用于 bge 类模型
        """
        if instruction is not None:
            data["instruction"] = instruction

        super().__init__(**data)
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

        if device:
            self.model = self.model.to(device)

    def _get_query_embedding(self, text: str) -> list[float]:
        # 使用指令前缀
        text = self.instruction + text if self.instruction else text
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    async def _aget_query_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)
