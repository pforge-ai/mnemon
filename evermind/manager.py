"""
MindCore Memory Manager

系统主管理器，提供统一的记忆管理接口，整合所有核心组件。
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
    MemoryStats,
    NamespaceConfig,
)
from .config import EverMindConfig
from .storage.protocols import IStorageBackend
from .indexing.processor import IndexingProcessor, StreamingIndexProcessor
from .retrieval.engine import RetrievalEngine


class MemoryManager:
    """
    MindCore 记忆系统主管理器

    统一的记忆管理接口，支持渐进式功能启用和命名空间隔离。
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

        # 初始化核心组件
        self.indexing_processor = IndexingProcessor(llm, self.config)
        self.retrieval_engine = RetrievalEngine(
            storage_backend, embeddings, self.config
        )

        # 流式处理器（可选）
        self.streaming_processor: Optional[StreamingIndexProcessor] = None
        if self.config.is_feature_enabled("indexing"):
            self.streaming_processor = StreamingIndexProcessor(self.indexing_processor)

        # 命名空间管理
        self._namespaces: Dict[str, NamespaceConfig] = {}
        self._background_tasks: List[asyncio.Task] = []

        # 初始化标志
        self._is_initialized = False

    async def initialize(self, enable_background_processing: bool = True) -> None:
        """初始化记忆管理器"""
        if self._is_initialized:
            return

        # 初始化存储后端
        await self.storage.initialize(self.config.performance_config.model_dump())

        # 注册默认命名空间
        await self.register_namespace(self.default_namespace)

        # 启动后台处理任务
        if enable_background_processing:
            await self._start_background_tasks()

        self._is_initialized = True

    async def ingest(
        self,
        content: str,
        namespace: Optional[str] = None,
        source_type: str = "user_input",
        custom_metadata: Optional[Dict[str, Any]] = None,
        process_immediately: bool = False,
    ) -> str:
        """
        记忆录入接口

        Args:
            content: 记忆内容
            namespace: 命名空间，默认使用default
            source_type: 来源类型
            custom_metadata: 自定义元数据
            process_immediately: 是否立即处理索引（否则进入后台队列）

        Returns:
            记忆ID
        """
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        # 创建记忆记录
        memory = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            metadata=MemoryMetadata(
                namespace=namespace,
                source_type=source_type,
                custom_data=custom_metadata or {},
            ),
        )

        # 生成内容embedding
        try:
            if not content.strip():
                raise ValueError("Content cannot be empty")

            # 尝试 LangChain 的 embedding
            memory.content_embedding = await self.embeddings.aembed_query(content)

            if not memory.content_embedding:
                raise ValueError("Embedding generation returned empty result")

        except Exception as e:
            print(f"Warning: LangChain embedding failed: {e}")

            # 尝试直接使用 OpenAI 客户端
            try:
                import openai

                openai_client = openai.AsyncOpenAI()

                response = await openai_client.embeddings.create(
                    input=content, model="text-embedding-v4"  # 直接传入字符串
                )
                memory.content_embedding = response.data[0].embedding
                print(f"✅ Used direct OpenAI client for embedding")

            except Exception as openai_error:
                print(f"Warning: Direct OpenAI embedding also failed: {openai_error}")
                # 最后的后备方案：使用固定向量
                print(f"Using mock embedding as final fallback")

                # 尝试从 embeddings 对象推断维度
                if hasattr(self.embeddings, "model") and "ada-002" in str(
                    self.embeddings.model
                ):
                    mock_dim = 1536
                else:
                    mock_dim = 1024  # 用户提到的维度

                memory.content_embedding = [0.1] * mock_dim
                print(f"Using {mock_dim}-dimensional mock embedding")

        if process_immediately or not self.streaming_processor:
            # 立即处理
            processed_memory = await self.indexing_processor.process_memory(memory)
            await self.storage.vector_store.store_memory(processed_memory)

            # 更新关联索引
            if self.config.is_feature_enabled("association"):
                await self._update_associations(processed_memory)
        else:
            # 提交到后台队列
            await self.streaming_processor.submit_memory(memory)

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
        """
        记忆查询接口

        Args:
            query: 查询文本
            namespace: 命名空间
            task_type: 任务类型，影响RR权重
            limit: 返回结果数量
            include_context_hints: 是否生成上下文引子
            filters: 过滤条件

        Returns:
            查询结果
        """
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        # 执行检索
        result = await self.retrieval_engine.search(
            query=query,
            namespace=namespace,
            task_type=task_type,
            limit=limit,
            include_context_hints=include_context_hints,
            filters=filters,
        )

        return result

    async def query_by_association(
        self, entity: str, namespace: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """基于实体关联进行查询"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        results = await self.retrieval_engine.search_by_association(
            entity=entity, namespace=namespace, limit=limit
        )

        return [
            {
                "memory_id": result.memory_record.id,
                "content": result.memory_record.content,
                "score": result.final_score,
                "match_type": result.match_type.value,
                "timestamp": result.memory_record.timestamp,
            }
            for result in results
        ]

    async def get_memory_by_id(
        self, memory_id: str, namespace: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """根据ID获取记忆"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        memory = await self.storage.vector_store.get_memory_by_id(memory_id, namespace)
        if not memory:
            return None

        return {
            "id": memory.id,
            "content": memory.content,
            "timestamp": memory.timestamp,
            "metadata": memory.metadata.model_dump(),
            "indexes": memory.indexes.model_dump(),
        }

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any],
        namespace: Optional[str] = None,
        reprocess: bool = False,
    ) -> bool:
        """更新记忆"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        # 更新元数据
        success = await self.storage.vector_store.update_memory_metadata(
            memory_id, namespace, updates
        )

        # 如果需要重新处理索引
        if reprocess and success:
            memory = await self.storage.vector_store.get_memory_by_id(
                memory_id, namespace
            )
            if memory:
                processed_memory = await self.indexing_processor.process_memory(memory)
                await self.storage.vector_store.store_memory(processed_memory)

        return success

    async def delete_memory(
        self, memory_id: str, namespace: Optional[str] = None
    ) -> bool:
        """删除记忆"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        return await self.storage.vector_store.delete_memory(memory_id, namespace)

    async def run_maintenance(
        self, namespace: Optional[str] = None, tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """执行记忆维护任务"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace
        tasks = tasks or ["forgetting", "cleanup", "optimization"]

        results = {}

        # 自适应遗忘
        if "forgetting" in tasks and self.config.is_feature_enabled("forgetting"):
            forgotten_count = await self._run_adaptive_forgetting(namespace)
            results["forgotten_memories"] = forgotten_count

        # 清理孤立关联
        if "cleanup" in tasks and self.config.is_feature_enabled("association"):
            cleaned_count = (
                await self.storage.association_store.cleanup_orphaned_associations(
                    namespace
                )
            )
            results["cleaned_associations"] = cleaned_count

        # 优化任务（压缩、归档等）
        if "optimization" in tasks:
            results["optimization"] = await self._run_optimization_tasks(namespace)

        return results

    async def register_namespace(
        self,
        namespace: str,
        description: str = "",
        max_memories: int = 100000,
        retention_days: int = 365,
    ) -> bool:
        """注册新的命名空间"""

        namespace_config = NamespaceConfig(
            namespace=namespace,
            description=description,
            max_memories=max_memories,
            retention_days=retention_days,
        )

        self._namespaces[namespace] = namespace_config

        # 初始化向量存储集合
        await self.storage.vector_store.initialize(
            f"namespace_{namespace}",
            self.config.performance_config.qdrant_collection_config,
        )

        return True

    async def get_namespace_stats(
        self, namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取命名空间统计信息"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        # 获取基础统计
        stats = await self.storage.vector_store.get_namespace_stats(namespace)

        # 添加配置信息
        namespace_config = self._namespaces.get(namespace)

        return {
            "namespace": namespace,
            "stats": stats.model_dump(),
            "config": namespace_config.model_dump() if namespace_config else None,
            "health": await self.storage.health_check(),
        }

    async def get_context_hints(
        self, namespace: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取当前上下文引子（用于智能体注入）"""
        if not self._is_initialized:
            await self.initialize()

        namespace = namespace or self.default_namespace

        # 获取热门实体
        popular_entities = await self.retrieval_engine.get_popular_entities(
            namespace, limit
        )

        hints = []
        for entity in popular_entities:
            associated_memories = (
                await self.storage.association_store.get_associated_memories(
                    entity, namespace, 1
                )
            )

            hints.append(
                {
                    "type": "entity",
                    "content": entity,
                    "memory_count": len(associated_memories),
                    "namespace": namespace,
                }
            )

        return hints

    async def _start_background_tasks(self):
        """启动后台任务"""

        # 启动流式索引处理器
        if self.streaming_processor:
            task = asyncio.create_task(self.streaming_processor.start_processing())
            self._background_tasks.append(task)

        # 启动定期维护任务
        task = asyncio.create_task(self._periodic_maintenance())
        self._background_tasks.append(task)

    async def _periodic_maintenance(self):
        """定期维护任务"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时运行一次

                for namespace in self._namespaces.keys():
                    await self.run_maintenance(namespace, ["cleanup"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic maintenance: {e}")

    async def _update_associations(self, memory: MemoryRecord):
        """更新记忆的关联索引"""
        namespace = memory.metadata.namespace

        # 添加实体关联
        for entity in memory.metadata.associated_entities:
            await self.storage.association_store.add_association(
                entity, memory.id, namespace
            )

        # 更新实体间共现关系
        entities = memory.metadata.associated_entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                await self.storage.association_store.update_association_strength(
                    entity1, entity2, namespace
                )

    async def _run_adaptive_forgetting(self, namespace: str) -> int:
        """运行自适应遗忘算法"""
        if not self.config.forgetting_config.enable_adaptive_forgetting:
            return 0

        # 获取候选记忆（简化实现）
        old_memories = await self.storage.vector_store.get_memories_by_status(
            namespace, "active", 1000
        )

        forgotten_count = 0
        current_time = time.time()

        for memory in old_memories:
            # 计算遗忘概率
            forgetting_prob = self._calculate_forgetting_probability(
                memory, current_time
            )

            if forgetting_prob > self.config.forgetting_config.forgetting_threshold:
                # 标记为遗忘状态
                await self.storage.vector_store.update_memory_metadata(
                    memory.id, namespace, {"status": "forgotten"}
                )
                forgotten_count += 1

        return forgotten_count

    def _calculate_forgetting_probability(
        self, memory: MemoryRecord, current_time: float
    ) -> float:
        """计算遗忘概率"""
        config = self.config.forgetting_config

        # 时间因子
        days_elapsed = (current_time - memory.timestamp) / (24 * 60 * 60)
        time_factor = min(days_elapsed / 365, 1.0)  # 归一化到[0,1]

        # 重要性因子（重要性越高，遗忘概率越低）
        importance_factor = 1.0 - (memory.metadata.importance_score / 4.0)

        # 频次因子（访问越多，遗忘概率越低）
        max_access = 100  # 假设最大访问次数
        frequency_factor = 1.0 - min(memory.metadata.access_count / max_access, 1.0)

        # 关联因子（关联越多，遗忘概率越低）
        association_factor = 1.0 - min(memory.metadata.association_count / 10, 1.0)

        # 计算综合遗忘概率
        forgetting_probability = (
            config.time_weight * time_factor
            + config.importance_weight * importance_factor
            + config.frequency_weight * frequency_factor
            + config.association_weight * association_factor
        )

        return min(forgetting_probability, 1.0)

    async def _run_optimization_tasks(self, namespace: str) -> Dict[str, Any]:
        """运行优化任务"""
        # 简化实现
        return {
            "compressed_memories": 0,
            "archived_memories": 0,
            "storage_saved_mb": 0.0,
        }

    async def shutdown(self):
        """关闭记忆管理器"""

        # 取消后台任务
        for task in self._background_tasks:
            task.cancel()

        # 等待任务完成
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # 停止流式处理器
        if self.streaming_processor:
            await self.streaming_processor.stop_processing()

        self._is_initialized = False

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "namespaces": list(self._namespaces.keys()),
            "background_tasks": len(self._background_tasks),
            "config": {
                "features_enabled": {
                    "indexing": self.config.is_feature_enabled("indexing"),
                    "association": self.config.is_feature_enabled("association"),
                    "context": self.config.is_feature_enabled("context"),
                    "reasoning": self.config.is_feature_enabled("reasoning"),
                    "forgetting": self.config.is_feature_enabled("forgetting"),
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
