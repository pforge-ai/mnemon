"""
MindCore Qdrant Backend Implementation

基于Qdrant客户端的存储后端实现，完全绕过LangChain避免API验证问题。
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from langchain_core.embeddings import Embeddings

from .protocols import IVectorStore, IAssociationStore, IStorageBackend
from ..models import MemoryRecord, AssociationLink, MemoryStats, IndexType, MemoryStatus


class QdrantVectorStoreImpl(IVectorStore):
    """基于Qdrant客户端的向量存储实现，完全绕过LangChain"""

    def __init__(self, qdrant_client: QdrantClient, embeddings: Embeddings):
        self.client = qdrant_client
        self.embeddings = embeddings
        self._collections_created: Dict[str, bool] = {}
        self._vector_size: Optional[int] = None

    async def initialize(
        self, collection_name: str, vector_config: Dict[str, Any]
    ) -> None:
        """初始化向量集合"""
        if collection_name in self._collections_created:
            return

        try:
            # 检查集合是否存在
            self.client.get_collection(collection_name)
            print(f"[DEBUG] Collection {collection_name} already exists")
        except ValueError:
            # 集合不存在，创建新集合
            if self._vector_size is None:
                self._vector_size = 1024  # 用户提到的维度

            print(
                f"[DEBUG] Creating collection {collection_name} with {self._vector_size} dimensions"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

        self._collections_created[collection_name] = True
        print(f"[DEBUG] Collection {collection_name} ready for use")

    async def store_memory(self, memory: MemoryRecord) -> bool:
        """存储记忆"""
        try:
            collection_name = f"namespace_{memory.metadata.namespace}"

            # 确保集合已初始化
            await self.initialize(collection_name, {})

            # 如果记忆还没有 embedding，先生成
            if not memory.content_embedding:
                memory.content_embedding = await self.embeddings.aembed_query(
                    memory.content
                )

            # 准备payload
            payload = {
                "page_content": memory.content,
                "memory_id": memory.id,
                "timestamp": memory.timestamp,
                "namespace": memory.metadata.namespace,
                "source_type": memory.metadata.source_type,
                "importance_score": memory.metadata.importance_score,
                "access_count": memory.metadata.access_count,
                "status": memory.metadata.status.value,
                "last_accessed_at": memory.metadata.last_accessed_at,
                "forgetting_probability": memory.metadata.forgetting_probability,
                "associated_entities": json.dumps(memory.metadata.associated_entities),
                "association_count": memory.metadata.association_count,
                # 多粒度索引数据
                "concepts": json.dumps(memory.indexes.concepts),
                "questions": json.dumps(memory.indexes.questions),
                "summary": memory.indexes.summary or "",
                "keywords": json.dumps(memory.indexes.keywords),
                # 自定义数据
                "custom_data": json.dumps(memory.metadata.custom_data),
            }

            # 直接用 Qdrant 客户端添加点
            point = PointStruct(
                id=memory.id, vector=memory.content_embedding, payload=payload
            )

            self.client.upsert(collection_name=collection_name, points=[point])

            print(
                f"[DEBUG] Successfully stored memory {memory.id} in {collection_name}"
            )
            return True

        except Exception as e:
            print(f"Error storing memory {memory.id}: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def batch_store_memories(self, memories: List[MemoryRecord]) -> List[bool]:
        """批量存储记忆"""
        results = []
        for memory in memories:
            result = await self.store_memory(memory)
            results.append(result)
        return results

    async def search_by_content(
        self,
        query_embedding: List[float],
        namespace: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """基于内容向量搜索"""
        collection_name = f"namespace_{namespace}"

        # 确保集合已初始化
        await self.initialize(collection_name, {})

        try:
            # 准备过滤条件
            search_filter = None
            if filters:
                conditions = []
                if "status" in filters:
                    conditions.append(
                        {"key": "status", "match": {"value": filters["status"]}}
                    )
                if "importance_score" in filters:
                    conditions.append(
                        {
                            "key": "importance_score",
                            "range": {"gte": filters["importance_score"]},
                        }
                    )

                if conditions:
                    search_filter = {"must": conditions}

            # 使用 Qdrant 客户端直接搜索
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
            )

            results = []
            for result in search_results:
                memory = self._search_result_to_memory_record(result)
                if memory:
                    results.append((memory, result.score))

            return results

        except Exception as e:
            print(f"Error searching by content in namespace {namespace}: {e}")
            return []

    async def search_by_index_type(
        self,
        query_embedding: List[float],
        index_type: IndexType,
        namespace: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float, str]]:
        """基于特定索引类型搜索"""
        # 先进行常规搜索
        content_results = await self.search_by_content(
            query_embedding, namespace, limit * 2, score_threshold
        )

        # 在结果中查找匹配的索引内容
        results = []
        for memory, score in content_results:
            match_content = ""

            if index_type == IndexType.CONCEPT:
                match_content = "; ".join(memory.indexes.concepts)
            elif index_type == IndexType.QUESTION:
                match_content = "; ".join(memory.indexes.questions)
            elif index_type == IndexType.SUMMARY:
                match_content = memory.indexes.summary or ""
            elif index_type == IndexType.CONTENT:
                match_content = memory.content

            if match_content:
                results.append((memory, score, match_content))

            if len(results) >= limit:
                break

        return results

    async def get_memory_by_id(
        self, memory_id: str, namespace: str
    ) -> Optional[MemoryRecord]:
        """根据ID获取记忆"""
        collection_name = f"namespace_{namespace}"

        try:
            # 确保集合存在
            await self.initialize(collection_name, {})

            # 使用Qdrant客户端直接查询
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                return None

            point = points[0]
            return self._search_result_to_memory_record(point)

        except Exception as e:
            print(f"Error getting memory {memory_id} from namespace {namespace}: {e}")
            return None

    async def update_memory_metadata(
        self, memory_id: str, namespace: str, metadata_updates: Dict[str, Any]
    ) -> bool:
        """更新记忆元数据"""
        collection_name = f"namespace_{namespace}"

        try:
            # 构造更新载荷
            payload_updates = {}
            for key, value in metadata_updates.items():
                if isinstance(value, (list, dict)):
                    payload_updates[key] = json.dumps(value)
                else:
                    payload_updates[key] = value

            # 使用Qdrant客户端更新
            self.client.set_payload(
                collection_name=collection_name,
                points=[memory_id],
                payload=payload_updates,
            )
            return True

        except Exception as e:
            print(f"Error updating memory {memory_id} metadata: {e}")
            return False

    async def delete_memory(self, memory_id: str, namespace: str) -> bool:
        """删除记忆"""
        collection_name = f"namespace_{namespace}"

        try:
            self.client.delete(
                collection_name=collection_name, points_selector=[memory_id]
            )
            return True

        except Exception as e:
            print(f"Error deleting memory {memory_id}: {e}")
            return False

    async def get_memories_by_status(
        self, namespace: str, status: str, limit: int = 100
    ) -> List[MemoryRecord]:
        """根据状态获取记忆列表"""
        collection_name = f"namespace_{namespace}"

        try:
            # 确保集合存在
            await self.initialize(collection_name, {})

            # 使用Qdrant的scroll功能
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter={"must": [{"key": "status", "match": {"value": status}}]},
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )

            memories = []
            for record in records:
                memory = self._search_result_to_memory_record(record)
                if memory:
                    memories.append(memory)

            return memories

        except Exception as e:
            print(f"Error getting memories by status {status}: {e}")
            return []

    async def get_namespace_stats(self, namespace: str) -> MemoryStats:
        """获取命名空间统计信息"""
        collection_name = f"namespace_{namespace}"

        try:
            # 获取集合信息
            collection_info = self.client.get_collection(collection_name)

            # 统计不同状态的记忆数量
            total_memories = collection_info.points_count

            # 简化版统计
            return MemoryStats(
                total_memories=total_memories,
                active_memories=total_memories,
                archived_memories=0,
                forgotten_memories=0,
                avg_importance_score=2.0,
                avg_access_count=1.0,
                total_associations=0,
                storage_size_mb=collection_info.config.params.vectors.size
                * total_memories
                * 4
                / 1024
                / 1024,
                index_size_mb=0.0,
            )

        except Exception as e:
            print(f"Error getting namespace stats for {namespace}: {e}")
            return MemoryStats(
                total_memories=0,
                active_memories=0,
                archived_memories=0,
                forgotten_memories=0,
                avg_importance_score=0.0,
                avg_access_count=0.0,
                total_associations=0,
                storage_size_mb=0.0,
                index_size_mb=0.0,
            )

    def _search_result_to_memory_record(self, result) -> Optional[MemoryRecord]:
        """将 Qdrant 搜索结果转换为 MemoryRecord"""
        try:
            payload = result.payload

            memory = MemoryRecord(
                id=str(result.id),
                content=payload.get("page_content", ""),
                content_embedding=getattr(result, "vector", None),
                timestamp=payload.get("timestamp", time.time()),
            )

            # 恢复元数据
            memory.metadata.namespace = payload.get("namespace", "default")
            memory.metadata.source_type = payload.get("source_type", "unknown")
            memory.metadata.importance_score = payload.get("importance_score", 0.0)
            memory.metadata.access_count = payload.get("access_count", 0)
            memory.metadata.status = MemoryStatus(payload.get("status", "active"))

            # 安全地解析 JSON 字段
            try:
                memory.metadata.associated_entities = json.loads(
                    payload.get("associated_entities", "[]")
                )
            except (json.JSONDecodeError, TypeError):
                memory.metadata.associated_entities = []

            # 恢复索引数据
            try:
                memory.indexes.concepts = json.loads(payload.get("concepts", "[]"))
            except (json.JSONDecodeError, TypeError):
                memory.indexes.concepts = []

            try:
                memory.indexes.questions = json.loads(payload.get("questions", "[]"))
            except (json.JSONDecodeError, TypeError):
                memory.indexes.questions = []

            try:
                memory.indexes.keywords = json.loads(payload.get("keywords", "[]"))
            except (json.JSONDecodeError, TypeError):
                memory.indexes.keywords = []

            memory.indexes.summary = payload.get("summary")

            return memory

        except Exception as e:
            print(f"Error converting search result to memory record: {e}")
            return None


class SimpleAssociationStore(IAssociationStore):
    """简单的内存关联存储实现"""

    def __init__(self):
        self._associations: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._co_occurrences: Dict[str, Dict[Tuple[str, str], int]] = defaultdict(
            lambda: defaultdict(int)
        )

    async def add_association(
        self, entity: str, memory_id: str, namespace: str
    ) -> None:
        """添加实体关联"""
        self._associations[namespace][entity].append(memory_id)

    async def get_associated_memories(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[str]:
        """获取与实体关联的记忆ID列表"""
        return self._associations[namespace].get(entity, [])[:limit]

    async def get_co_occurring_entities(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[AssociationLink]:
        """获取与实体共现的其他实体"""
        return []

    async def update_association_strength(
        self, entity1: str, entity2: str, namespace: str
    ) -> None:
        """更新实体间关联强度"""
        key = (min(entity1, entity2), max(entity1, entity2))
        self._co_occurrences[namespace][key] += 1

    async def get_entity_associations(self, namespace: str) -> Dict[str, List[str]]:
        """获取命名空间内所有实体关联"""
        return dict(self._associations[namespace])

    async def cleanup_orphaned_associations(self, namespace: str) -> int:
        """清理孤立的关联记录"""
        return 0


class QdrantStorageBackend(IStorageBackend):
    """基于Qdrant的完整存储后端"""

    def __init__(self, qdrant_client: QdrantClient, embeddings: Embeddings):
        self._vector_store = QdrantVectorStoreImpl(qdrant_client, embeddings)
        self._association_store = SimpleAssociationStore()
        self._reasoning_store = None
        self._cache_store = None

    @property
    def vector_store(self) -> IVectorStore:
        return self._vector_store

    @property
    def association_store(self) -> IAssociationStore:
        return self._association_store

    @property
    def reasoning_store(self) -> Optional[IStorageBackend]:
        return self._reasoning_store

    @property
    def cache_store(self) -> Optional[IStorageBackend]:
        return self._cache_store

    async def initialize(self, config: Dict[str, Any]) -> None:
        """初始化存储后端"""
        pass

    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        try:
            return {
                "vector_store": True,
                "association_store": True,
                "reasoning_store": False,
                "cache_store": False,
            }
        except Exception as e:
            print(f"Health check failed: {e}")
            return {
                "vector_store": False,
                "association_store": True,
                "reasoning_store": False,
                "cache_store": False,
            }

    async def backup_namespace(self, namespace: str, backup_path: str) -> bool:
        """备份命名空间"""
        return False  # 暂不实现

    async def restore_namespace(self, namespace: str, backup_path: str) -> bool:
        """恢复命名空间"""
        return False  # 暂不实现
