"""
MindCore Qdrant Backend Implementation

基于Qdrant客户端的存储后端实现，完全绕过LangChain避免API验证问题。
"""

import json
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus
from qdrant_client import QdrantClient, models
from langchain_core.embeddings import Embeddings

from .protocols import IVectorStore, IAssociationStore, IStorageBackend

# 修复：导入缺失的 MemoryMetadata 和 MultiGranularIndex 类
from ..models import (
    MemoryRecord,
    AssociationLink,
    MemoryStats,
    IndexType,
    MemoryStatus,
    MemoryMetadata,
    MultiGranularIndex,
)


class QdrantVectorStoreImpl(IVectorStore):
    """基于Qdrant客户端的向量存储实现"""

    def __init__(self, qdrant_client: QdrantClient, embeddings: Embeddings):
        self.client = qdrant_client
        self.embeddings = embeddings
        self._collections_created = set()

    async def initialize(
        self, collection_name: str, vector_config: Dict[str, Any]
    ) -> None:
        if collection_name in self._collections_created:
            return
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_config.get("vectors", {}).get("size", 1024),
                    distance=Distance.COSINE,
                ),
            )
        self._collections_created.add(collection_name)

    def _memory_to_payload(self, memory: MemoryRecord) -> Dict[str, Any]:
        """将 MemoryRecord 转换为 Qdrant payload，处理复杂类型"""
        payload = memory.metadata.model_dump()
        payload.update(memory.indexes.model_dump())

        payload["content"] = memory.content
        payload["timestamp"] = memory.timestamp
        payload["status"] = payload["status"].value

        for key, value in payload.items():
            if isinstance(value, (list, dict)):
                payload[key] = json.dumps(value)

        return payload

    async def store_memory(self, memory: MemoryRecord) -> bool:
        """存储单条记忆"""
        collection_name = f"namespace_{memory.metadata.namespace}"
        try:
            await self.initialize(collection_name, {})

            if not memory.content_embedding:
                raise ValueError("Content embedding is missing.")

            payload = self._memory_to_payload(memory)
            point = PointStruct(
                id=memory.id, vector=memory.content_embedding, payload=payload
            )

            response = self.client.upsert(
                collection_name=collection_name, points=[point], wait=True
            )
            return response.status == UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Error storing memory {memory.id}: {e}")
            traceback.print_exc()
            return False

    async def batch_store_memories(self, memories: List[MemoryRecord]) -> List[bool]:
        """批量存储记忆"""
        if not memories:
            return []

        memories_by_namespace = defaultdict(list)
        for mem in memories:
            memories_by_namespace[mem.metadata.namespace].append(mem)

        results = {}
        for namespace, mem_list in memories_by_namespace.items():
            collection_name = f"namespace_{namespace}"
            points = [
                PointStruct(
                    id=mem.id,
                    vector=mem.content_embedding,
                    payload=self._memory_to_payload(mem),
                )
                for mem in mem_list
                if mem.content_embedding
            ]

            try:
                if points:
                    self.client.upsert(
                        collection_name=collection_name, points=points, wait=True
                    )
                for mem in mem_list:
                    results[mem.id] = True
            except Exception as e:
                print(f"Error batch storing memories in {namespace}: {e}")
                for mem in mem_list:
                    results[mem.id] = False

        return [results.get(mem.id, False) for mem in memories]

    def _point_to_memory_record(
        self, point: models.ScoredPoint or models.Record
    ) -> Optional[MemoryRecord]:
        """将 Qdrant Point 转换为 MemoryRecord"""
        try:
            payload = point.payload

            def _safe_json_load(field_name: str, default_value):
                raw = payload.get(field_name)
                if isinstance(raw, str):
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return default_value
                return raw if raw is not None else default_value

            return MemoryRecord(
                id=str(point.id),
                content=payload.get("content", ""),
                content_embedding=getattr(point, "vector", None),
                timestamp=payload.get("timestamp", time.time()),
                metadata=MemoryMetadata(
                    namespace=payload.get("namespace", "default"),
                    source_type=payload.get("source_type", "unknown"),
                    importance_score=payload.get("importance_score", 0.0),
                    access_count=payload.get("access_count", 0),
                    status=MemoryStatus(payload.get("status", "active")),
                    last_accessed_at=payload.get("last_accessed_at", time.time()),
                    forgetting_probability=payload.get("forgetting_probability", 0.0),
                    associated_entities=_safe_json_load("associated_entities", []),
                    association_count=payload.get("association_count", 0),
                    custom_data=_safe_json_load("custom_data", {}),
                ),
                indexes=MultiGranularIndex(
                    concepts=_safe_json_load("concepts", []),
                    questions=_safe_json_load("questions", []),
                    summary=payload.get("summary"),
                    keywords=_safe_json_load("keywords", []),
                ),
            )
        except Exception as e:
            print(
                f"Error converting point {getattr(point, 'id', 'N/A')} to memory record: {e}"
            )
            return None

    async def search_by_content(
        self,
        query_embedding: List[float],
        namespace: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        collection_name = f"namespace_{namespace}"
        await self.initialize(collection_name, {})
        search_filter = models.Filter(**filters) if filters else None
        try:
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
            )
            return [
                (mem, result.score)
                for result in search_results
                if (mem := self._point_to_memory_record(result)) is not None
            ]
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
        content_results = await self.search_by_content(
            query_embedding, namespace, limit * 2, score_threshold
        )
        results = []
        for memory, score in content_results:
            match_content = ""
            if index_type == IndexType.CONCEPT:
                match_content = "; ".join(memory.indexes.concepts)
            elif index_type == IndexType.QUESTION:
                match_content = "; ".join(memory.indexes.questions)
            elif index_type == IndexType.SUMMARY:
                match_content = memory.indexes.summary or ""
            if match_content:
                results.append((memory, score, match_content))
        return results[:limit]

    async def get_memory_by_id(
        self, memory_id: str, namespace: str
    ) -> Optional[MemoryRecord]:
        collection_name = f"namespace_{namespace}"
        try:
            await self.initialize(collection_name, {})
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=True,
            )
            return self._point_to_memory_record(points[0]) if points else None
        except Exception as e:
            print(f"Error getting memory {memory_id} from {namespace}: {e}")
            return None

    async def update_memory_metadata(
        self, memory_id: str, namespace: str, metadata_updates: Dict[str, Any]
    ) -> bool:
        collection_name = f"namespace_{namespace}"
        try:
            payload = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in metadata_updates.items()
            }
            self.client.set_payload(
                collection_name=collection_name,
                points=[memory_id],
                payload=payload,
                wait=True,
            )
            return True
        except Exception as e:
            print(f"Error updating memory {memory_id} metadata: {e}")
            return False

    async def delete_memory(self, memory_id: str, namespace: str) -> bool:
        collection_name = f"namespace_{namespace}"
        try:
            await self.initialize(collection_name, {})
            result = self.client.delete(
                collection_name=collection_name, points_selector=[memory_id], wait=True
            )
            return result.status == UpdateStatus.COMPLETED
        except Exception as e:
            print(f"Error deleting memory {memory_id}: {e}")
            return False

    async def get_memories_by_status(
        self, namespace: str, status: str, limit: int = 100
    ) -> List[MemoryRecord]:
        collection_name = f"namespace_{namespace}"
        try:
            await self.initialize(collection_name, {})
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="status", match=models.MatchValue(value=status)
                    )
                ]
            )
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            return [
                mem for rec in records if (mem := self._point_to_memory_record(rec))
            ]
        except Exception as e:
            print(f"Error getting memories by status {status}: {e}")
            return []

    async def get_namespace_stats(self, namespace: str) -> MemoryStats:
        collection_name = f"namespace_{namespace}"
        try:
            collection_info = self.client.get_collection(collection_name)
            vector_size = collection_info.config.params.vectors.size
            points_count = collection_info.points_count
            storage_mb = (vector_size * points_count * 4) / (1024 * 1024)
            return MemoryStats(
                total_memories=points_count,
                active_memories=points_count,
                storage_size_mb=storage_mb,
                archived_memories=0,
                forgotten_memories=0,
                index_size_mb=0.0,
                avg_importance_score=0.0,
                avg_access_count=0.0,
                total_associations=0,
            )
        except Exception as e:
            print(f"Error getting namespace stats for {namespace}: {e}")
            return MemoryStats(
                total_memories=0,
                active_memories=0,
                archived_memories=0,
                forgotten_memories=0,
                storage_size_mb=0.0,
                index_size_mb=0.0,
                avg_importance_score=0.0,
                avg_access_count=0.0,
                total_associations=0,
            )


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
        if memory_id not in self._associations[namespace][entity]:
            self._associations[namespace][entity].append(memory_id)

    async def get_associated_memories(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[str]:
        return self._associations[namespace].get(entity, [])[:limit]

    async def get_association_count(self, entity: str, namespace: str) -> int:
        """获取实体关联的记忆数量"""
        return len(self._associations[namespace].get(entity, []))

    async def get_co_occurring_entities(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[AssociationLink]:
        return []

    async def update_association_strength(
        self, entity1: str, entity2: str, namespace: str
    ) -> None:
        key = tuple(sorted((entity1, entity2)))
        self._co_occurrences[namespace][key] += 1

    async def get_entity_associations(self, namespace: str) -> Dict[str, List[str]]:
        return dict(self._associations.get(namespace, {}))

    async def cleanup_orphaned_associations(self, namespace: str) -> int:
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
        pass

    async def health_check(self) -> Dict[str, bool]:
        return {
            "vector_store": True,
            "association_store": True,
            "reasoning_store": self._reasoning_store is not None,
            "cache_store": self._cache_store is not None,
        }

    async def backup_namespace(self, namespace: str, backup_path: str) -> bool:
        return False

    async def restore_namespace(self, namespace: str, backup_path: str) -> bool:
        return False
