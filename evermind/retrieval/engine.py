"""
EverMind Retrieval Engine

基于RR权重的智能检索引擎，支持多粒度检索和上下文引子生成。
"""

import time
import math
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.embeddings import Embeddings

from ..models import (
    MemoryRecord,
    RetrievedMemory,
    QueryResult,
    ContextHint,
    IndexType,
)
from ..config import EverMindConfig, RRWeights
from ..storage.protocols import IStorageBackend


class RetrievalEngine:
    """基于RR权重的检索引擎"""

    def __init__(
        self,
        storage_backend: IStorageBackend,
        embeddings: Embeddings,
        config: EverMindConfig,
    ):
        self.storage = storage_backend
        self.embeddings = embeddings
        self.config = config
        self._query_embedding_cache: Dict[str, List[float]] = {}

    async def search(
        self,
        query: str,
        namespace: str,
        task_type: str = "default",
        limit: int = 10,
        include_context_hints: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """执行智能检索"""
        start_time = time.time()
        weights = self.config.get_task_weights(task_type)
        query_embedding = await self._get_query_embedding(query)

        if not query_embedding:
            return QueryResult(
                retrieved_memories=[],
                context_hints=[],
                total_matches=0,
                processing_time_ms=0,
            )

        raw_results = await self._multi_granular_search(
            query_embedding, namespace, limit * 2, filters
        )
        ranked_results = self._rank_by_rr_weights(raw_results, weights)
        final_results = ranked_results[:limit]

        await self._update_access_stats(final_results)

        context_hints = []
        if include_context_hints and self.config.is_feature_enabled("context"):
            context_hints = await self._generate_context_hints(final_results)

        processing_time = (time.time() - start_time) * 1000
        return QueryResult(
            retrieved_memories=final_results,
            context_hints=context_hints,
            total_matches=len(raw_results),
            processing_time_ms=processing_time,
        )

    async def search_by_association(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[RetrievedMemory]:
        """基于实体关联进行检索"""
        associated_memory_ids = (
            await self.storage.association_store.get_associated_memories(
                entity, namespace, limit * 2
            )
        )
        if not associated_memory_ids:
            return []

        memories = [
            mem
            for mem_id in associated_memory_ids
            if (
                mem := await self.storage.vector_store.get_memory_by_id(
                    mem_id, namespace
                )
            )
        ]

        def sort_key(memory: MemoryRecord) -> float:
            return (
                memory.metadata.access_count * 0.3
                + memory.metadata.importance_score * 0.7
            )

        memories.sort(key=sort_key, reverse=True)

        results = []
        for memory in memories[:limit]:
            results.append(
                RetrievedMemory(
                    memory_record=memory,
                    relevance_score=1.0,
                    recency_score=self._calculate_recency_score(memory.timestamp),
                    final_score=sort_key(memory),
                    match_type=IndexType.CONTENT,
                    match_content=f"关联实体: {entity}",
                )
            )
        return results

    async def _multi_granular_search(
        self,
        query_embedding: List[float],
        namespace: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Tuple[MemoryRecord, float, IndexType, str]]:
        """执行多粒度检索"""
        tasks = [
            self.storage.vector_store.search_by_content(
                query_embedding, namespace, limit, 0.0, filters
            )
        ]
        if self.config.is_feature_enabled("indexing"):
            tasks.append(
                self.storage.vector_store.search_by_index_type(
                    query_embedding, IndexType.CONCEPT, namespace, limit // 2
                )
            )
            tasks.append(
                self.storage.vector_store.search_by_index_type(
                    query_embedding, IndexType.QUESTION, namespace, limit // 2
                )
            )

        results = await asyncio.gather(*tasks)
        content_results = results[0]
        concept_results = results[1] if len(results) > 1 else []
        question_results = results[2] if len(results) > 2 else []

        all_results_map: Dict[str, Tuple[MemoryRecord, float, IndexType, str]] = {}

        def add_to_map(search_results, index_type, score_multiplier=1.0):
            for item in search_results:
                memory, score = item[:2]
                match_content = item[2] if len(item) > 2 else memory.content[:100]
                if (
                    memory.id not in all_results_map
                    or score * score_multiplier > all_results_map[memory.id][1]
                ):
                    all_results_map[memory.id] = (
                        memory,
                        score * score_multiplier,
                        index_type,
                        match_content,
                    )

        add_to_map(content_results, IndexType.CONTENT)
        add_to_map(concept_results, IndexType.CONCEPT, 0.8)
        add_to_map(question_results, IndexType.QUESTION, 0.9)

        return list(all_results_map.values())

    def _rank_by_rr_weights(
        self,
        raw_results: List[Tuple[MemoryRecord, float, IndexType, str]],
        weights: RRWeights,
    ) -> List[RetrievedMemory]:
        """使用RR权重计算最终排序"""
        ranked_memories = []
        for memory, relevance_score, match_type, match_content in raw_results:
            recency_score = self._calculate_recency_score(memory.timestamp)
            final_score = (
                weights.relevance * relevance_score + weights.recency * recency_score
            )
            ranked_memories.append(
                RetrievedMemory(
                    memory_record=memory,
                    relevance_score=relevance_score,
                    recency_score=recency_score,
                    final_score=final_score,
                    match_type=match_type,
                    match_content=match_content,
                )
            )
        ranked_memories.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_memories

    async def _generate_context_hints(
        self, retrieved_memories: List[RetrievedMemory]
    ) -> List[ContextHint]:
        """生成上下文引子"""
        hints = []
        all_entities = set()
        for mem in retrieved_memories:
            all_entities.update(mem.memory_record.metadata.associated_entities)
            for question in mem.memory_record.indexes.questions[:2]:
                hints.append(
                    ContextHint(
                        type=IndexType.QUESTION,
                        content=f"可回答: {question}",
                        associated_memory_count=1,
                        importance_level="low",
                    )
                )

        for entity in list(all_entities)[:5]:
            count = await self.storage.association_store.get_association_count(
                entity, retrieved_memories[0].memory_record.metadata.namespace
            )
            if count > 0:
                hints.append(
                    ContextHint(
                        type=IndexType.CONCEPT,
                        content=f"实体: {entity}",
                        associated_memory_count=count,
                        importance_level="high" if count >= 5 else "medium",
                    )
                )

        hints.sort(key=lambda x: x.associated_memory_count, reverse=True)
        return hints[: self.config.performance_config.max_context_memories]

    async def _update_access_stats(self, retrieved_memories: List[RetrievedMemory]):
        """更新访问统计"""
        for mem in retrieved_memories:
            await self.storage.vector_store.update_memory_metadata(
                mem.memory_record.id,
                mem.memory_record.metadata.namespace,
                {
                    "access_count": mem.memory_record.metadata.access_count + 1,
                    "last_accessed_at": time.time(),
                },
            )

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """获取查询的embedding（带缓存）"""
        if query in self._query_embedding_cache:
            return self._query_embedding_cache[query]
        embedding = await self.embeddings.aembed_query(query)
        if embedding and len(self._query_embedding_cache) > 100:
            self._query_embedding_cache.pop(next(iter(self._query_embedding_cache)))
        if embedding:
            self._query_embedding_cache[query] = embedding
        return embedding

    def _calculate_recency_score(self, timestamp: float) -> float:
        """计算时效性分数"""
        days_elapsed = (time.time() - timestamp) / 86400
        return math.exp(-self.config.if_config.recency_decay_rate * days_elapsed)

    async def get_popular_entities(self, namespace: str, limit: int = 10) -> List[str]:
        """获取热门实体"""
        entity_associations = (
            await self.storage.association_store.get_entity_associations(namespace)
        )
        entity_counts = [
            (entity, len(mem_ids)) for entity, mem_ids in entity_associations.items()
        ]
        entity_counts.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, count in entity_counts[:limit]]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            "query_cache_size": len(self._query_embedding_cache),
            "enabled_features": {
                "multi_granular_indexing": self.config.is_feature_enabled("indexing"),
                "association_tracking": self.config.is_feature_enabled("association"),
                "context_injection": self.config.is_feature_enabled("context"),
            },
        }
