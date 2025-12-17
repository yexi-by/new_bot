from pathlib import Path
import aiofiles
from siliconflow_embedding import SiliconFlowEmbedding
import httpx
import asyncio


async def read_txt_file(file_str: str) -> str:
    """合并txt文件"""
    file_path = Path(file_str)
    results = file_path.rglob("*.txt")
    all_data = []
    for file in results:
        async with aiofiles.open(file, mode="r",encoding="utf-8") as f:
            data = await f.read()
            all_data.append(data)
    combined_text = "\n".join(all_data)
    return combined_text

def split_text(text:str):
    result="".join(text.split())
    return result

if __name__ == "__main__":
    client = httpx.AsyncClient()
    se = SiliconFlowEmbedding(
        client=client,
        apk_key="sk-bskcjrcbjmbkcsbovbjkydelvegydkbbebonpgkxlnuybtac",
        base_url="https://api.siliconflow.cn/v1/embeddings",
    )
    async def main():
        file_str=input("输入路径")
        combined_text= await read_txt_file(file_str)
        result=split_text(combined_text)
        output_file = "debug_output.txt"
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(result)
    asyncio.run(main())
    
