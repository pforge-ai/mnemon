"""
EverMind: 渐进式智能记忆系统

为智能体提供类人记忆能力，支持多粒度索引、RR权重检索、上下文引子生成等功能。
"""

__version__ = "0.2.0"
__author__ = "EverMind Team"

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

# 核心组件导入
from .config import EverMindConfig, RRWeights, IFConfig, ForgettingConfig
from .models import (
    MemoryRecord,
    MemoryMetadata,
    QueryResult,
    RetrievedMemory,
    ContextHint,
    MultiGranularIndex,
    MemoryStats,
)
from .manager import MemoryManager
from .storage.qdrant_backend import QdrantStorageBackend
from .indexing.processor import IndexingProcessor
from .retrieval.engine import RetrievalEngine

# 便捷的公共API
__all__ = [
    # 核心类
    "MemoryManager",
    "EverMindConfig",
    # 配置类
    "RRWeights",
    "IFConfig",
    "ForgettingConfig",
    # 数据模型
    "MemoryRecord",
    "MemoryMetadata",
    "QueryResult",
    "RetrievedMemory",
    "ContextHint",
    "MultiGranularIndex",
    "MemoryStats",
    # 组件类
    "IndexingProcessor",
    "RetrievalEngine",
    "QdrantStorageBackend",
    # 便捷函数
    "create_memory_manager",
    "create_simple_memory_manager",
]


def create_memory_manager(
    qdrant_client: QdrantClient,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    config: Optional[EverMindConfig] = None,
    namespace: str = "default",
) -> MemoryManager:
    """
    创建完整功能的记忆管理器

    Args:
        qdrant_client: Qdrant客户端实例
        llm: 语言模型实例
        embeddings: 向量化模型实例
        config: 配置对象，None时使用默认配置
        namespace: 默认命名空间

    Returns:
        配置好的MemoryManager实例

    Example:
        ```python
        from qdrant_client import QdrantClient
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import evermind

        # 初始化依赖组件
        client = QdrantClient(":memory:")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        embeddings = OpenAIEmbeddings()

        # 创建记忆管理器
        memory_manager = evermind.create_memory_manager(
            qdrant_client=client,
            llm=llm,
            embeddings=embeddings,
            namespace="my_agent"
        )

        # 初始化并使用
        await memory_manager.initialize()
        memory_id = await memory_manager.ingest("今天天气很好")
        result = await memory_manager.query("天气怎么样？")
        ```
    """
    if config is None:
        config = EverMindConfig()

    # 创建存储后端
    storage_backend = QdrantStorageBackend(qdrant_client, embeddings)

    # 创建记忆管理器
    manager = MemoryManager(
        storage_backend=storage_backend,
        llm=llm,
        embeddings=embeddings,
        config=config,
        default_namespace=namespace,
    )

    return manager


def create_simple_memory_manager(
    qdrant_client: QdrantClient,
    llm: BaseLanguageModel,
    embeddings: Embeddings,
    namespace: str = "default",
) -> MemoryManager:
    """
    创建简化版记忆管理器（适合快速上手）

    默认配置：
    - 启用基础多粒度索引
    - 启用关联追踪
    - 启用上下文引子
    - 禁用推理链和自适应遗忘

    Args:
        qdrant_client: Qdrant客户端实例
        llm: 语言模型实例
        embeddings: 向量化模型实例
        namespace: 默认命名空间

    Returns:
        配置好的简化版MemoryManager实例

    Example:
        ```python
        from qdrant_client import QdrantClient
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        import evermind

        # 快速创建
        client = QdrantClient(":memory:")
        llm = ChatOpenAI()
        embeddings = OpenAIEmbeddings()

        memory_manager = evermind.create_simple_memory_manager(
            client, llm, embeddings, "demo"
        )

        await memory_manager.initialize()
        ```
    """

    # 创建简化配置
    from .config import IndexingConfig

    simple_config = EverMindConfig(
        # 核心功能：启用
        enable_multi_granular_indexing=True,
        enable_association_tracking=True,
        enable_context_injection=True,
        # 高级功能：禁用（简化使用）
        enable_reasoning_chains=False,
        enable_adaptive_forgetting=False,
        # 调整阈值使其更容易触发
        indexing_config=IndexingConfig(
            concept_extraction_threshold=1.5,  # 降低阈值
            question_extraction_threshold=2.0,
            max_concepts_per_memory=3,  # 减少数量
            max_questions_per_memory=2,
        ),
    )

    return create_memory_manager(
        qdrant_client, llm, embeddings, simple_config, namespace
    )


# 版本信息和功能特性
FEATURES = {
    "multi_granular_indexing": "多粒度索引（概念/问题/摘要）",
    "rr_weighted_retrieval": "RR权重检索（相关性+时效性）",
    "context_hints": "上下文引子生成",
    "association_tracking": "实体关联追踪",
    "adaptive_forgetting": "自适应遗忘机制",
    "reasoning_chains": "轻量级推理链",
    "namespace_isolation": "命名空间隔离",
    "streaming_processing": "流式后台处理",
}


def get_version_info() -> Dict[str, Any]:
    """获取版本和功能信息"""
    return {
        "version": __version__,
        "author": __author__,
        "features": FEATURES,
        "dependencies": {
            "langchain_core": ">=0.2.0",
            "qdrant_client": ">=1.7.0",
            "pydantic": ">=2.0.0",
        },
    }


def print_welcome():
    """打印欢迎信息"""
    print(
        f"""
🧠 EverMind v{__version__} 
渐进式智能记忆系统

核心特性:
• 多粒度索引：概念/问题/摘要自动抽取
• RR权重检索：相关性+时效性智能排序  
• 上下文引子：为智能体提供记忆线索
• 关联追踪：实体关系自动发现
• 命名空间隔离：多用户/多场景支持

快速开始:
  memory_manager = evermind.create_simple_memory_manager(client, llm, embeddings)
  await memory_manager.initialize()
  
详细文档: https://github.com/pforge-ai/evermind
    """
    )


# 可选的自动欢迎信息
import os

if os.getenv("EVERMIND_SHOW_WELCOME", "").lower() == "true":
    print_welcome()
