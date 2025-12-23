"""向量搜索模块，提供基于 FAISS 的向量相似度搜索功能。"""

import asyncio
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .siliconflow_embedding import SiliconFlowEmbedding


class SearchVectors:
    """基于 FAISS 的向量搜索类。

    该类用于加载预构建的 FAISS 索引和 ID 映射文件，
    并提供向量相似度搜索功能。

    Attributes:
        id_mapping: ID 映射列表，将 FAISS 索引位置映射到实际的文档 ID。
        index: FAISS 索引对象，用于高效的向量相似度搜索。
    """

    id_mapping: list[str]
    index: faiss.Index

    def __init__(self, directory: str) -> None:
        """初始化向量搜索实例。

        Args:
            directory: 包含 index.faiss 和 id_mapping.json 文件的目录路径。
        """
        self.id_mapping, self.index = self._load_index_and_mapping(directory=directory)

    def _load_index_and_mapping(self, directory: str) -> tuple[list[str], faiss.Index]:
        """从指定目录加载 FAISS 索引和 ID 映射。"""
        index_path = Path(directory) / "index.faiss"
        mapping_path = Path(directory) / "id_mapping.json"
        with open(mapping_path, "r", encoding="utf-8") as f:
            content = f.read()
            id_mapping: list[str] = json.loads(content)
        with open(index_path, "rb") as f:
            index_bytes = f.read()
            index: faiss.Index = faiss.deserialize_index(
                np.frombuffer(index_bytes, dtype=np.uint8)
            )
        return id_mapping, index

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[str]:
        """根据查询向量搜索最相似的文档。

        对查询向量进行 L2 归一化后，使用 FAISS 索引进行相似度搜索。

        Args:
            query_vector: 查询向量，浮点数列表。
            top_k: 返回的最相似结果数量。

        Returns:
            匹配的文档 ID 列表，按相似度降序排列。
        """
        query_np = np.array([query_vector], dtype="float32")
        await asyncio.to_thread(faiss.normalize_L2, query_np)
        distances: np.ndarray
        indices: np.ndarray
        distances, indices = await asyncio.to_thread(self.index.search, query_np, top_k)  # type: ignore
        results = [self.id_mapping[idx] for idx in indices[0] if idx != -1]
        return results

    @staticmethod
    def _extract_embedding(result: dict[str, Any]) -> list[float]:
        """从 API 响应中提取嵌入向量。"""
        data: list[dict[str, Any]] = result.get("data", [])
        embedding: list[float] = data[0]["embedding"]
        return embedding

    async def search_by_text(
        self,
        siliconflow_embedding: SiliconFlowEmbedding,
        query_text: str,
        model: str,
        top_k: int = 5,
    ) -> list[str]:
        """根据文本查询搜索最相似的文档。

        先将文本转换为嵌入向量，然后进行向量相似度搜索。

        Args:
            siliconflow_embedding: SiliconFlow Embedding 客户端实例。
            query_text: 查询文本。
            model: 用于生成嵌入向量的模型名称。
            top_k: 返回的最相似结果数量，默认为 5。

        Returns:
            匹配的文档 ID 列表，按相似度降序排列。
        """
        result = await siliconflow_embedding.get_embedding(
            text=[query_text],
            model=model,
        )
        vector = self._extract_embedding(result=result)
        matched_ids = await self.search(query_vector=vector, top_k=top_k)
        return matched_ids
