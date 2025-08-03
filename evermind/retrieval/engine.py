"""
EverMind Retrieval Engine

基于RR权重的智能检索引擎，支持多粒度检索和上下文引子生成。
"""

import time
import math
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.embeddings import Embeddings

from ..models import (
    MemoryRecord,
    RetrievedMemory,
    QueryResult,
    ContextHint,
    IndexType,
    AssociationLink,
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

        # 缓存查询embedding以提高性能
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

        # 1. 获取任务类型对应的权重配置
        weights = self.config.get_task_weights(task_type)

        # 2. 生成查询embedding
        query_embedding = await self._get_query_embedding(query)

        # 3. 执行多粒度检索
        raw_results = await self._multi_granular_search(
            query_embedding, namespace, limit * 2, filters
        )

        # 4. 计算RR综合分数并排序
        ranked_results = await self._rank_by_rr_weights(raw_results, weights)

        # 5. 截取最终结果
        final_results = ranked_results[:limit]

        # 6. 更新访问统计
        await self._update_access_stats(final_results)

        # 7. 生成上下文引子（如果需要）
        context_hints = []
        if include_context_hints:
            context_hints = await self._generate_context_hints(
                query, namespace, final_results
            )

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

        # 获取关联的记忆ID
        associated_memory_ids = (
            await self.storage.association_store.get_associated_memories(
                entity, namespace, limit * 2
            )
        )

        if not associated_memory_ids:
            return []

        # 获取完整的记忆记录
        memories = []
        for memory_id in associated_memory_ids:
            memory = await self.storage.vector_store.get_memory_by_id(
                memory_id, namespace
            )
            if memory:
                memories.append(memory)

        # 按访问频次和重要性排序
        def sort_key(memory: MemoryRecord) -> float:
            return (
                memory.metadata.access_count * 0.3
                + memory.metadata.importance_score * 0.7
            )

        memories.sort(key=sort_key, reverse=True)

        # 转换为RetrievedMemory格式
        results = []
        for memory in memories[:limit]:
            retrieved_memory = RetrievedMemory(
                memory_record=memory,
                relevance_score=1.0,  # 关联检索的相关性设为1
                recency_score=self._calculate_recency_score(memory.timestamp),
                final_score=sort_key(memory),
                match_type=IndexType.CONTENT,
                match_content=f"关联实体: {entity}",
            )
            results.append(retrieved_memory)

        return results

    async def _multi_granular_search(
        self,
        query_embedding: List[float],
        namespace: str,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float, IndexType, str]]:
        """执行多粒度检索"""

        all_results = []

        # 1. 内容级检索（主要）
        content_results = await self.storage.vector_store.search_by_content(
            query_embedding, namespace, limit, 0.0, filters
        )

        for memory, score in content_results:
            all_results.append((memory, score, IndexType.CONTENT, memory.content[:100]))

        # 2. 概念级检索
        if self.config.is_feature_enabled("indexing"):
            concept_results = await self.storage.vector_store.search_by_index_type(
                query_embedding, IndexType.CONCEPT, namespace, limit // 2
            )

            for memory, score, match_content in concept_results:
                all_results.append(
                    (memory, score * 0.8, IndexType.CONCEPT, match_content)
                )

        # 3. 问题级检索
        if self.config.is_feature_enabled("indexing"):
            question_results = await self.storage.vector_store.search_by_index_type(
                query_embedding, IndexType.QUESTION, namespace, limit // 2
            )

            for memory, score, match_content in question_results:
                all_results.append(
                    (memory, score * 0.9, IndexType.QUESTION, match_content)
                )

        # 去重（同一个记忆可能在多个粒度中匹配）
        seen_memory_ids = set()
        unique_results = []

        for memory, score, index_type, match_content in all_results:
            if memory.id not in seen_memory_ids:
                seen_memory_ids.add(memory.id)
                unique_results.append((memory, score, index_type, match_content))

        return unique_results

    async def _rank_by_rr_weights(
        self,
        raw_results: List[Tuple[MemoryRecord, float, IndexType, str]],
        weights: RRWeights,
    ) -> List[RetrievedMemory]:
        """使用RR权重计算最终排序"""

        ranked_memories = []
        current_time = time.time()

        for memory, relevance_score, match_type, match_content in raw_results:

            # 计算时效性分数（指数衰减）
            days_elapsed = (current_time - memory.timestamp) / (24 * 60 * 60)
            recency_score = math.exp(
                -self.config.if_config.recency_decay_rate * days_elapsed
            )

            # 计算最终RR分数
            final_score = (
                weights.relevance * relevance_score + weights.recency * recency_score
            )

            retrieved_memory = RetrievedMemory(
                memory_record=memory,
                relevance_score=relevance_score,
                recency_score=recency_score,
                final_score=final_score,
                match_type=match_type,
                match_content=match_content,
            )

            ranked_memories.append(retrieved_memory)

        # 按最终分数排序
        ranked_memories.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_memories

    async def _generate_context_hints(
        self, query: str, namespace: str, retrieved_memories: List[RetrievedMemory]
    ) -> List[ContextHint]:
        """生成上下文引子"""

        if not self.config.is_feature_enabled("context"):
            return []

        hints = []

        # 1. 从检索结果中收集概念
        all_concepts = set()
        all_entities = set()

        for retrieved_memory in retrieved_memories:
            memory = retrieved_memory.memory_record
            all_concepts.update(memory.indexes.concepts)
            all_entities.update(memory.metadata.associated_entities)

        # 2. 生成概念级引子
        if all_concepts:
            concept_hints = self._create_hints_from_concepts(
                list(all_concepts)[:5], namespace
            )
            hints.extend(concept_hints)

        # 3. 生成实体级引子
        if all_entities:
            entity_hints = await self._create_hints_from_entities(
                list(all_entities)[:5], namespace
            )
            hints.extend(entity_hints)

        # 4. 生成问题级引子
        question_hints = self._create_hints_from_questions(retrieved_memories)
        hints.extend(question_hints)

        # 按重要程度排序，限制数量
        hints.sort(key=lambda x: x.associated_memory_count, reverse=True)
        return hints[: self.config.performance_config.max_context_memories]

    def _create_hints_from_concepts(
        self, concepts: List[str], namespace: str
    ) -> List[ContextHint]:
        """从概念创建引子"""
        hints = []

        for concept in concepts:
            hint = ContextHint(
                type=IndexType.CONCEPT,
                content=f"概念: {concept}",
                associated_memory_count=1,  # 简化实现
                importance_level="medium",
            )
            hints.append(hint)

        return hints

    async def _create_hints_from_entities(
        self, entities: List[str], namespace: str
    ) -> List[ContextHint]:
        """从实体创建引子"""
        hints = []

        for entity in entities:
            # 获取实体的关联记忆数量
            associated_memories = (
                await self.storage.association_store.get_associated_memories(
                    entity, namespace, 100
                )
            )

            if len(associated_memories) > 0:
                importance_level = "high" if len(associated_memories) >= 5 else "medium"

                hint = ContextHint(
                    type=IndexType.CONCEPT,
                    content=f"实体: {entity}",
                    associated_memory_count=len(associated_memories),
                    importance_level=importance_level,
                )
                hints.append(hint)

        return hints

    def _create_hints_from_questions(
        self, retrieved_memories: List[RetrievedMemory]
    ) -> List[ContextHint]:
        """从问题创建引子"""
        hints = []

        for retrieved_memory in retrieved_memories:
            memory = retrieved_memory.memory_record

            for question in memory.indexes.questions[:2]:  # 每个记忆最多2个问题
                hint = ContextHint(
                    type=IndexType.QUESTION,
                    content=f"可回答: {question}",
                    associated_memory_count=1,
                    importance_level="low",
                )
                hints.append(hint)

        return hints

    async def _update_access_stats(self, retrieved_memories: List[RetrievedMemory]):
        """更新访问统计"""

        for retrieved_memory in retrieved_memories:
            memory = retrieved_memory.memory_record
            memory_id = memory.id
            namespace = memory.metadata.namespace

            # 更新访问计数和最后访问时间
            await self.storage.vector_store.update_memory_metadata(
                memory_id,
                namespace,
                {
                    "access_count": memory.metadata.access_count + 1,
                    "last_accessed_at": time.time(),
                },
            )

    async def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询的embedding（带缓存）"""

        if query in self._query_embedding_cache:
            return self._query_embedding_cache[query]

        embedding = await self.embeddings.aembed_query(query)

        # 简单的LRU缓存，限制大小
        if len(self._query_embedding_cache) > 100:
            # 删除最老的一个
            oldest_key = next(iter(self._query_embedding_cache))
            del self._query_embedding_cache[oldest_key]

        self._query_embedding_cache[query] = embedding
        return embedding

    def _calculate_recency_score(self, timestamp: float) -> float:
        """计算时效性分数"""
        current_time = time.time()
        days_elapsed = (current_time - timestamp) / (24 * 60 * 60)
        return math.exp(-self.config.if_config.recency_decay_rate * days_elapsed)

    async def get_popular_entities(self, namespace: str, limit: int = 10) -> List[str]:
        """获取热门实体（用于引子生成）"""

        entity_associations = (
            await self.storage.association_store.get_entity_associations(namespace)
        )

        # 按关联记忆数量排序
        entity_counts = [
            (entity, len(memory_ids))
            for entity, memory_ids in entity_associations.items()
        ]
        entity_counts.sort(key=lambda x: x[1], reverse=True)

        return [entity for entity, count in entity_counts[:limit]]

    async def search_similar_to_memory(
        self, memory_id: str, namespace: str, limit: int = 5
    ) -> List[RetrievedMemory]:
        """查找与指定记忆相似的其他记忆"""

        # 获取原始记忆
        original_memory = await self.storage.vector_store.get_memory_by_id(
            memory_id, namespace
        )
        if not original_memory or not original_memory.content_embedding:
            return []

        # 使用原始记忆的embedding进行搜索
        results = await self.storage.vector_store.search_by_content(
            original_memory.content_embedding,
            namespace,
            limit + 1,  # +1 因为会包含原始记忆本身
            0.5,  # 设置相似度阈值
        )

        # 过滤掉原始记忆本身
        similar_memories = []
        weights = self.config.get_task_weights("default")

        for memory, relevance_score in results:
            if memory.id != memory_id:
                recency_score = self._calculate_recency_score(memory.timestamp)
                final_score = (
                    weights.relevance * relevance_score
                    + weights.recency * recency_score
                )

                retrieved_memory = RetrievedMemory(
                    memory_record=memory,
                    relevance_score=relevance_score,
                    recency_score=recency_score,
                    final_score=final_score,
                    match_type=IndexType.CONTENT,
                    match_content=memory.content[:100],
                )
                similar_memories.append(retrieved_memory)

        return similar_memories[:limit]

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
