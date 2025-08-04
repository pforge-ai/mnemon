"""
MindCore Memory Manager

系统主管理器，提供统一的记忆管理接口，整合所有核心组件。
重构：完全面向LangChain核心协议(BaseLanguageModel, Embeddings)编程，实现真正的厂商解耦。
"""

import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from .models import (
    MemoryRecord,
    MemoryMetadata,
    QueryResult,
    NamespaceConfig,
)
from .config import EverMindConfig
from .storage.protocols import IStorageBackend
from .indexing.processor import IndexingProcessor, StreamingIndexProcessor
from .retrieval.engine import RetrievalEngine


class MemoryManager:
    """
    MindCore 记忆系统主管理器
    """

    def __init__(
        self,
        storage_backend: IStorageBackend,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        config: Optional[EverMindConfig] = None,
        default_namespace: str = "default",
    ):
        self.storage = storage_backend
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or EverMindConfig()
        self.default_namespace = default_namespace

        self.indexing_processor = IndexingProcessor(llm, self.config)
        self.retrieval_engine = RetrievalEngine(
            storage_backend, embeddings, self.config
        )

        self.streaming_processor: Optional[StreamingIndexProcessor] = None
        if self.config.is_feature_enabled("indexing"):
            self.streaming_processor = StreamingIndexProcessor(
                self.indexing_processor, self.storage, self.config
            )

        self._namespaces: Dict[str, NamespaceConfig] = {}
        self._is_initialized = False

    async def initialize(self, enable_background_processing: bool = True) -> None:
        """初始化记忆管理器"""
        if self._is_initialized:
            return
        await self.storage.initialize({})
        await self.register_namespace(self.default_namespace)
        if enable_background_processing and self.streaming_processor:
            self.streaming_processor.start_processing()
        self._is_initialized = True

    async def _generate_embedding(self, content: str) -> Optional[List[float]]:
        """为内容生成向量，完全依赖传入的 Embeddings 协议。"""
        if not content.strip():
            print("Warning: Content is empty, skipping embedding.")
            return None
        try:
            return await self.embeddings.aembed_query(content)
        except Exception as e:
            print(
                f"Error: The provided embedding model failed to process the request: {e}"
            )
            return None

    async def ingest(
        self,
        content: str,
        namespace: Optional[str] = None,
        source_type: str = "user_input",
        custom_metadata: Optional[Dict[str, Any]] = None,
        process_immediately: bool = False,
    ) -> str:
        """记忆录入接口"""
        await self.initialize()
        namespace = namespace or self.default_namespace

        memory = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            metadata=MemoryMetadata(
                namespace=namespace,
                source_type=source_type,
                custom_data=custom_metadata or {},
            ),
        )

        memory.content_embedding = await self._generate_embedding(content)
        if not memory.content_embedding:
            raise ValueError(
                f"Failed to generate embedding for the content. Please check your embedding model provider's configuration and status."
            )

        if self.streaming_processor and not process_immediately:
            await self.storage.vector_store.store_memory(memory)
            await self.streaming_processor.submit_memory(memory)
        else:
            processed_memory = await self.indexing_processor.process_memory(memory)
            await self.storage.vector_store.store_memory(processed_memory)
            if self.config.is_feature_enabled("association"):
                await self._update_associations(processed_memory)
        return memory.id

    async def query(
        self,
        query: str,
        namespace: Optional[str] = None,
        task_type: str = "default",
        limit: int = 10,
        include_context_hints: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """记忆查询接口"""
        await self.initialize()
        namespace = namespace or self.default_namespace
        return await self.retrieval_engine.search(
            query, namespace, task_type, limit, include_context_hints, filters
        )

    async def query_by_association(
        self, entity: str, namespace: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """基于实体关联进行查询"""
        await self.initialize()
        namespace = namespace or self.default_namespace
        results = await self.retrieval_engine.search_by_association(
            entity=entity, namespace=namespace, limit=limit
        )
        # 格式化输出以匹配 example.py 的期望
        response = []
        for res in results:
            response.append(
                {
                    "memory_id": res.memory_record.id,
                    "content": res.memory_record.content,
                    "score": res.final_score,
                    "match_type": res.match_type.value,
                    "timestamp": res.memory_record.timestamp,
                }
            )
        return response

    async def get_memory_by_id(
        self, memory_id: str, namespace: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """根据ID获取记忆"""
        await self.initialize()
        namespace = namespace or self.default_namespace
        memory = await self.storage.vector_store.get_memory_by_id(memory_id, namespace)
        return memory.model_dump() if memory else None

    async def get_context_hints(
        self, namespace: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取当前上下文引子（用于智能体注入）"""
        await self.initialize()
        namespace = namespace or self.default_namespace
        popular_entities = await self.retrieval_engine.get_popular_entities(
            namespace, limit
        )
        hints = []
        for entity in popular_entities:
            # 仅获取关联数量，无需完整记录
            count = await self.storage.association_store.get_association_count(
                entity, namespace
            )
            hints.append({"type": "entity", "content": entity, "memory_count": count})
        return hints

    async def get_namespace_stats(
        self, namespace: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """获取命名空间统计信息"""
        await self.initialize()
        namespace = namespace or self.default_namespace
        if namespace not in self._namespaces:
            return None
        stats = await self.storage.vector_store.get_namespace_stats(namespace)
        return {
            "namespace": namespace,
            "stats": stats.model_dump(),
            "config": self._namespaces[namespace].model_dump(),
            "health": await self.storage.health_check(),
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        is_streaming_active = (
            self.streaming_processor._is_running if self.streaming_processor else False
        )
        return {
            "namespaces": list(self._namespaces.keys()),
            "background_tasks": 1 if is_streaming_active else 0,
            "config": {
                "features_enabled": {
                    name: self.config.is_feature_enabled(name)
                    for name in [
                        "indexing",
                        "association",
                        "context",
                        "reasoning",
                        "forgetting",
                    ]
                }
            },
            "processing_stats": {
                "indexing": self.indexing_processor.get_processing_stats(),
                "retrieval": self.retrieval_engine.get_retrieval_stats(),
                "streaming_queue_size": (
                    self.streaming_processor.get_queue_size()
                    if self.streaming_processor
                    else 0
                ),
            },
        }

    async def _update_associations(self, memory: MemoryRecord):
        namespace = memory.metadata.namespace
        entities = memory.metadata.associated_entities
        for entity in entities:
            await self.storage.association_store.add_association(
                entity, memory.id, namespace
            )
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                await self.storage.association_store.update_association_strength(
                    entity1, entity2, namespace
                )

    async def register_namespace(self, namespace: str, **kwargs) -> bool:
        if namespace not in self._namespaces:
            self._namespaces[namespace] = NamespaceConfig(namespace=namespace, **kwargs)
            collection_name = f"namespace_{namespace}"
            vector_config = self.config.performance_config.qdrant_collection_config
            await self.storage.vector_store.initialize(collection_name, vector_config)
        return True

    async def shutdown(self):
        if self.streaming_processor:
            await self.streaming_processor.stop_processing()
        self._is_initialized = False
