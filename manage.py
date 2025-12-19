import asyncio
import logging

import httpx

from config import EmbeddingConfig
from core.model.rag import (
    SiliconFlowEmbedding,
    async_process_pipeline,
    intelligent_split,
    read_txt_file,
    split_text,
)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到终端
    ],
)

if __name__ == "__main__":
    embedding_config = EmbeddingConfig(
        api_key="sk-bskcjrcbjmbkcsbovbjkydelvegydkbbebonpgkxlnuybtac",
        base_url="https://api.siliconflow.cn/v1/embeddings",
        model_name="Qwen/Qwen3-Embedding-8B",
        provider_type="siliconflow",
        retry_count=3,
        retry_delay=1,
    )
    client = httpx.AsyncClient()
    se = SiliconFlowEmbedding(client=client, embedding_config=embedding_config)

    async def main():
        file_str = input("输入路径")
        storage_path = input("输入储存路径")
        combined_text = await read_txt_file(file_str)
        result = await asyncio.to_thread(split_text, text=combined_text)
        chunks = await asyncio.to_thread(intelligent_split, text=result, chunk_size=200)
        await async_process_pipeline(
            chunks=chunks,
            max_lines=5,
            minute=1000,
            consumer_count=1000,
            siliconflow_embedding=se,
            folder_path=storage_path,
            model=embedding_config.model_name,
        )

    asyncio.run(main())
