import asyncio
import itertools
import json
from pathlib import Path

import aiofiles

from .base import VectorStorage
from .siliconflow_embedding import SiliconFlowEmbedding


async def read_txt_file(file_str: str) -> str:
    """合并txt文件"""
    file_path = Path(file_str)
    results = file_path.rglob("*.txt")
    all_data = []
    for file in results:
        async with aiofiles.open(file, mode="r", encoding="utf-8") as f:
            data = await f.read()
            all_data.append(data)
    combined_text = await asyncio.to_thread(lambda: "\n".join(all_data))
    return combined_text


def split_text(text: str) -> str:
    result = "".join(text.split())
    return result


def intelligent_split(text: str, chunk_size: int) -> list[str]:
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
        found_split = False
        for i in range(current, n):
            if (open_count == close_count) and (text[i] in delimiters):
                chunks.append(text[start : i + 1])
                end = i + 1
                found_split = True
                break
            if text[i] == close_quote:
                close_count += 1
                continue
            elif text[i] == open_quote:
                open_count += 1
                continue
        if not found_split:
            chunks.append(text[start:])
            break
        start = end
    return chunks


def get_vector_representation(result: dict, chunk: list[str]) -> list[VectorStorage]:
    vector_list = []
    data: list = result.get("data", [])
    if not data:
        raise ValueError("数据为空")
    data.sort(key=lambda x: x["index"])
    for k, v in zip(chunk, data):
        vector = VectorStorage(name=k, vectors=v["embedding"])
        vector_list.append(vector)
    return vector_list


async def producer(
    task_queue: asyncio.Queue[list[str]], chunks: list[str], max_lines: int
) -> None:
    it = iter(chunks)
    batch_count = 0
    total_items = 0
    while True:
        chunk = list(itertools.islice(it, max_lines))
        if not chunk:
            break
        await task_queue.put(chunk)
        batch_count += 1
        total_items += len(chunk)


async def token_dispenser(token_queue: asyncio.Queue[int], minute: int) -> None:
    interval = 60.0 / minute
    token_count = 0
    while True:
        await token_queue.put(1)
        token_count += 1
        await asyncio.sleep(interval)


async def consumer(
    task_queue: asyncio.Queue[list[str]],
    token_queue: asyncio.Queue[int],
    siliconflow_embedding: SiliconFlowEmbedding,
    result_queue: asyncio.Queue[list[VectorStorage]],
    model: str,
) -> None:
    processed_count = 0
    while True:
        chunk: list[str] = await task_queue.get()
        await token_queue.get()
        try:
            result = await siliconflow_embedding.get_embedding(text=chunk, model=model)

            vector_list = await asyncio.to_thread(
                get_vector_representation, result=result, chunk=chunk
            )

            await result_queue.put(vector_list)
            processed_count += len(chunk)
        except Exception:
            await task_queue.put(chunk)  # 重新放入队列重试
        task_queue.task_done()


async def write_data(
    folder_path: str,
    result_queue: asyncio.Queue[list[VectorStorage]],
) -> None:
    path = Path(folder_path)
    file_path = path / "vector.json"
    first_item = True
    total_items = 0
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write("{")
        while True:
            vector_list = await result_queue.get()
            if not vector_list:
                result_queue.task_done()
                break
            for item in vector_list:
                if not first_item:
                    await f.write(",")
                else:
                    first_item = False
                key_str = json.dumps(item.name, ensure_ascii=False)
                value_str = json.dumps(item.vectors)
                line = f"\n{key_str}:{value_str}"
                await f.write(line)
                total_items += 1
            result_queue.task_done()
        await f.write("\n}")


async def async_process_pipeline(
    chunks: list[str],
    max_lines: int,
    minute: int,
    consumer_count: int,
    siliconflow_embedding: SiliconFlowEmbedding,
    model: str,
    folder_path: str,
) -> None:
    task_queue = asyncio.Queue()
    token_queue = asyncio.Queue(maxsize=1)
    result_queue = asyncio.Queue()
    consumers: list[asyncio.Task] = []

    token_task = asyncio.create_task(
        token_dispenser(token_queue=token_queue, minute=minute)
    )
    for _ in range(consumer_count):
        con = asyncio.create_task(
            consumer(
                task_queue=task_queue,
                token_queue=token_queue,
                siliconflow_embedding=siliconflow_embedding,
                result_queue=result_queue,
                model=model,
            )
        )
        consumers.append(con)
    writer_task = asyncio.create_task(
        write_data(folder_path=folder_path, result_queue=result_queue)
    )
    producer_task = asyncio.create_task(
        producer(task_queue=task_queue, chunks=chunks, max_lines=max_lines)
    )
    await producer_task
    await task_queue.join()
    token_task.cancel()
    for c in consumers:
        c.cancel()
    await result_queue.put(None)
    await writer_task
