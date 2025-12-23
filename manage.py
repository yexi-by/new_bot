import asyncio
import httpx
from config import EmbeddingConfig
from core.model.rag import (
    SiliconFlowEmbedding,
    VectorizeConfig,
    vectorize_text,
)
from log import setup_exception_handler

setup_exception_handler()
embedding_config = EmbeddingConfig(
    api_key="sk-bskcjrcbjmbkcsbovbjkydelvegydkbbebonpgkxlnuybtac",
    base_url="https://api.siliconflow.cn/v1/embeddings",
    model_name="Qwen/Qwen3-Embedding-8B",
    provider_type="siliconflow",
    retry_count=20,
    retry_delay=5,
)
client = httpx.AsyncClient()
se = SiliconFlowEmbedding(client=client, embedding_config=embedding_config)
config = VectorizeConfig(
    tokens_per_minute=1000,
    consumer_count=1000,
)
async def main():
    folder_str = input("输入路径")
    await vectorize_text(
        folder_str=folder_str,
        vectorize_config=config,
        siliconflow_embedding=se,
        model=embedding_config.model_name,
    )
if __name__ == "__main__":
    asyncio.run(main())
