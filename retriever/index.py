import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from models.embeddings import BGEEmbedding


def build_index():
    # 使用 BGEEmbedding 作为全局 embedding 模型
    Settings.embed_model = BGEEmbedding(model_name=os.getenv('EMBEDDING_MODEL'), cache_dir="./data/cache")
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()
