"""
EverMind Storage Protocols

定义存储层的抽象接口，支持不同向量数据库的可插拔实现。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

from ..models import (
    MemoryRecord,
    RetrievedMemory,
    AssociationLink,
    MemoryStats,
    IndexType,
)


class IVectorStore(ABC):
    """向量存储抽象接口"""

    @abstractmethod
    async def initialize(
        self, collection_name: str, vector_config: Dict[str, Any]
    ) -> None:
        """初始化向量集合"""
        pass

    @abstractmethod
    async def store_memory(self, memory: MemoryRecord) -> bool:
        """存储单条记忆及其多粒度索引"""
        pass

    @abstractmethod
    async def batch_store_memories(self, memories: List[MemoryRecord]) -> List[bool]:
        """批量存储记忆"""
        pass

    @abstractmethod
    async def search_by_content(
        self,
        query_embedding: List[float],
        namespace: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """基于内容向量搜索"""
        pass

    @abstractmethod
    async def search_by_index_type(
        self,
        query_embedding: List[float],
        index_type: IndexType,
        namespace: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float, str]]:
        """基于特定索引类型搜索，返回匹配的具体内容"""
        pass

    @abstractmethod
    async def get_memory_by_id(
        self, memory_id: str, namespace: str
    ) -> Optional[MemoryRecord]:
        """根据ID获取记忆"""
        pass

    @abstractmethod
    async def update_memory_metadata(
        self, memory_id: str, namespace: str, metadata_updates: Dict[str, Any]
    ) -> bool:
        """更新记忆元数据"""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str, namespace: str) -> bool:
        """删除记忆"""
        pass

    @abstractmethod
    async def get_memories_by_status(
        self, namespace: str, status: str, limit: int = 100
    ) -> List[MemoryRecord]:
        """根据状态获取记忆列表"""
        pass

    @abstractmethod
    async def get_namespace_stats(self, namespace: str) -> MemoryStats:
        """获取命名空间统计信息"""
        pass


class IAssociationStore(ABC):
    """关联存储抽象接口"""

    @abstractmethod
    async def add_association(
        self, entity: str, memory_id: str, namespace: str
    ) -> None:
        """添加实体关联"""
        pass

    @abstractmethod
    async def get_associated_memories(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[str]:
        """获取与实体关联的记忆ID列表"""
        pass

    @abstractmethod
    async def get_co_occurring_entities(
        self, entity: str, namespace: str, limit: int = 10
    ) -> List[AssociationLink]:
        """获取与实体共现的其他实体"""
        pass

    @abstractmethod
    async def update_association_strength(
        self, entity1: str, entity2: str, namespace: str
    ) -> None:
        """更新实体间关联强度"""
        pass

    @abstractmethod
    async def get_entity_associations(self, namespace: str) -> Dict[str, List[str]]:
        """获取命名空间内所有实体关联"""
        pass

    @abstractmethod
    async def cleanup_orphaned_associations(self, namespace: str) -> int:
        """清理孤立的关联记录"""
        pass


class IReasoningStore(ABC):
    """推理链存储抽象接口"""

    @abstractmethod
    async def add_reasoning_chain(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relation_type: str,
        confidence: float,
        reasoning: str,
        namespace: str,
    ) -> bool:
        """添加推理链"""
        pass

    @abstractmethod
    async def get_reasoning_chains_from(
        self, memory_id: str, namespace: str
    ) -> List[Dict[str, Any]]:
        """获取从指定记忆出发的推理链"""
        pass

    @abstractmethod
    async def get_reasoning_chains_to(
        self, memory_id: str, namespace: str
    ) -> List[Dict[str, Any]]:
        """获取指向指定记忆的推理链"""
        pass

    @abstractmethod
    async def find_reasoning_path(
        self, from_memory_id: str, to_memory_id: str, namespace: str, max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """查找两个记忆间的推理路径"""
        pass


class ICacheStore(ABC):
    """缓存存储抽象接口"""

    @abstractmethod
    async def set_cache(
        self, key: str, value: Any, ttl_seconds: Optional[int] = None
    ) -> None:
        """设置缓存"""
        pass

    @abstractmethod
    async def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass

    @abstractmethod
    async def delete_cache(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    async def clear_namespace_cache(self, namespace: str) -> int:
        """清理命名空间缓存"""
        pass

    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        pass


class IStorageBackend(ABC):
    """完整存储后端接口，组合所有存储功能"""

    @property
    @abstractmethod
    def vector_store(self) -> IVectorStore:
        """向量存储"""
        pass

    @property
    @abstractmethod
    def association_store(self) -> IAssociationStore:
        """关联存储"""
        pass

    @property
    @abstractmethod
    def reasoning_store(self) -> Optional[IReasoningStore]:
        """推理链存储（可选）"""
        pass

    @property
    @abstractmethod
    def cache_store(self) -> Optional[ICacheStore]:
        """缓存存储（可选）"""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """初始化存储后端"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        pass

    @abstractmethod
    async def backup_namespace(self, namespace: str, backup_path: str) -> bool:
        """备份命名空间"""
        pass

    @abstractmethod
    async def restore_namespace(self, namespace: str, backup_path: str) -> bool:
        """恢复命名空间"""
        pass
