from config import EmbeddingConfig
import httpx
import aiofiles
from core.model.rag import SiliconFlowEmbedding,read_txt_file,split_text,intelligent_split
import asyncio

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
        combined_text = await read_txt_file(file_str)
        result = split_text(combined_text)
        chunks = intelligent_split(text=result, chunk_size=200)
        file_path="debug.txt"
        async with aiofiles.open(file_path,"w",encoding="utf-8") as f:
            for c in chunks:
                await f.write(f"{c}\n\n")

    asyncio.run(main())
    
