from pathlib import Path
import aiofiles
import asyncio
import itertools
from siliconflow_embedding import SiliconFlowEmbedding
from base import VectorStorage

async def read_txt_file(file_str: str) -> str:
    """合并txt文件"""
    file_path = Path(file_str)
    results = file_path.rglob("*.txt")
    all_data = []
    for file in results:
        async with aiofiles.open(file, mode="r", encoding="utf-8") as f:
            data = await f.read()
            all_data.append(data)
    combined_text = "\n".join(all_data)
    return combined_text


def split_text(text: str):
    result = "".join(text.split())
    return result


def intelligent_split(text: str, chunk_size: int) -> list:
    chunks = []
    start = 0
    end = 0
    n = len(text)
    delimiters = {"。", "！", "？"}
    open_quote = "“"
    close_quote = "”"
    while start < n:
        close_count = 0
        open_count = 0
        current = start + chunk_size
        if current > n:
            chunks.append(text[start:])
            break
        for i in range(start, current):
            if text[i] == open_quote:
                open_count += 1
            elif text[i] == close_quote:
                close_count += 1
        for i in range(current, n):
            if (open_count == close_count) and (text[i] in delimiters):
                chunks.append(text[start : i + 1])
                end = i + 1
                break
            if text[i] == close_quote:
                close_count += 1
                continue
            elif text[i] == open_quote:
                open_count += 1
                continue
        start = end
    return chunks
def get_vector_representation(result:dict,chunk:list[str]):
    vector_list=[]
    data:list=result.get("data",[])
    if not data:
        raise ValueError("数据为空")
    data.sort(key=lambda x:x["index"])
    for k,v in zip(chunk,data):
        vector=VectorStorage(
            name=k,
            vectors=v["embedding"]
        )
        vector_list.append(vector)
    return vector_list
    
        

async def producer(task_queue:asyncio.Queue,chunks:list[str],max_lines:int)->None:
    it = iter(chunks)
    while True:
        chunk=list(itertools.islice(it,max_lines))
        if not chunk:
            break
        await task_queue.put(chunk)

async def token_dispenser(token_queue:asyncio.Queue,minute:int):
    interval = 60.0 /minute
    while True:
        await token_queue.put(1)
        await asyncio.sleep(interval)
        
async def consumer(task_queue:asyncio.Queue,token_queue:asyncio.Queue,siliconflow_embedding:SiliconFlowEmbedding,model:str):
    while True:
        chunk=await task_queue.get()
        token=await token_queue.get()
        result= await siliconflow_embedding.get_embedding(text=chunk,model=model)
        
        
        