"""
EverMind Data Models

定义记忆系统的核心数据结构，支持多粒度索引和渐进式功能。
"""

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# === 基础枚举类型 ===


class MemoryStatus(str, Enum):
    """记忆状态枚举"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class IndexType(str, Enum):
    """索引类型枚举"""

    CONCEPT = "concept"
    QUESTION = "question"
    SUMMARY = "summary"
    CONTENT = "content"


class ReasoningRelationType(str, Enum):
    """推理关系类型枚举"""

    CAUSES = "causes"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SIMILAR_TO = "similar_to"


# === 核心数据模型 ===


class MemoryMetadata(BaseModel):
    """记忆元数据"""

    namespace: str = Field(description="命名空间/分片ID")
    source_type: str = Field(description="记忆来源类型")
    importance_score: float = Field(0.0, ge=0.0, le=4.0, description="重要性分数[0-4]")
    access_count: int = Field(0, ge=0, description="访问次数")
    status: MemoryStatus = Field(MemoryStatus.ACTIVE, description="记忆状态")

    # 遗忘相关
    last_accessed_at: float = Field(
        default_factory=time.time, description="最后访问时间"
    )
    forgetting_probability: float = Field(0.0, ge=0.0, le=1.0, description="遗忘概率")

    # 关联相关
    associated_entities: List[str] = Field(
        default_factory=list, description="关联实体列表"
    )
    association_count: int = Field(0, ge=0, description="关联密度")

    # 自定义数据
    custom_data: Dict[str, Any] = Field(
        default_factory=dict, description="自定义元数据"
    )


class MultiGranularIndex(BaseModel):
    """多粒度索引结构"""

    concepts: List[str] = Field(default_factory=list, description="抽取的概念列表")
    questions: List[str] = Field(default_factory=list, description="可回答的问题列表")
    summary: Optional[str] = Field(None, description="内容摘要")
    keywords: List[str] = Field(default_factory=list, description="关键词列表")


class ReasoningChain(BaseModel):
    """推理链节点"""

    from_memory_id: str = Field(description="源记忆ID")
    to_memory_id: str = Field(description="目标记忆ID")
    relation_type: ReasoningRelationType = Field(description="关系类型")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    reasoning: str = Field(description="推理过程描述")
    created_at: float = Field(default_factory=time.time, description="创建时间")


class MemoryRecord(BaseModel):
    """完整的记忆记录"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="全局唯一ID")
    content: str = Field(description="记忆内容")
    content_embedding: Optional[List[float]] = Field(None, description="内容向量")
    timestamp: float = Field(default_factory=time.time, description="创建时间戳")

    # 元数据和索引
    metadata: MemoryMetadata = Field(description="记忆元数据")
    indexes: MultiGranularIndex = Field(
        default_factory=MultiGranularIndex, description="多粒度索引"
    )

    # 推理链（可选）
    reasoning_chains: List[ReasoningChain] = Field(
        default_factory=list, description="推理链列表"
    )

    @property
    def memory_id_with_namespace(self) -> str:
        """返回带命名空间的完整ID"""
        return f"{self.metadata.namespace}:{self.id}"

    def update_access_info(self) -> None:
        """更新访问信息"""
        self.metadata.access_count += 1
        self.metadata.last_accessed_at = time.time()


class AssociationLink(BaseModel):
    """实体关联链接"""

    entity: str = Field(description="实体名称")
    memory_ids: List[str] = Field(description="包含该实体的记忆ID列表")
    co_occurrence_count: int = Field(0, description="共现次数")
    strength: float = Field(0.0, ge=0.0, le=1.0, description="关联强度")


# === 检索相关模型 ===


class RetrievedMemory(BaseModel):
    """检索到的记忆"""

    memory_record: MemoryRecord = Field(description="完整记忆记录")
    relevance_score: float = Field(description="相关性分数")
    recency_score: float = Field(description="时效性分数")
    final_score: float = Field(description="最终RR综合分数")
    match_type: IndexType = Field(description="匹配的索引类型")
    match_content: str = Field(description="匹配的具体内容")


class ContextHint(BaseModel):
    """上下文引子/线索"""

    type: IndexType = Field(description="引子类型")
    content: str = Field(description="引子内容")
    associated_memory_count: int = Field(description="关联记忆数量")
    importance_level: str = Field(description="重要程度: high/medium/low")


class QueryResult(BaseModel):
    """查询结果"""

    retrieved_memories: List[RetrievedMemory] = Field(description="检索到的记忆列表")
    context_hints: List[ContextHint] = Field(description="生成的上下文引子")
    total_matches: int = Field(description="总匹配数量")
    processing_time_ms: float = Field(description="处理耗时(毫秒)")


# === LLM结构化输出模型 ===


class ImportanceRating(BaseModel):
    """重要性评分结果"""

    score: float = Field(ge=0.0, le=4.0, description="重要性分数[0-4]")
    reasoning: str = Field(description="评分理由")
    extracted_entities: List[str] = Field(
        default_factory=list, description="抽取的实体"
    )


class ConceptExtraction(BaseModel):
    """概念抽取结果"""

    concepts: List[str] = Field(description="抽取的概念列表")
    keywords: List[str] = Field(description="关键词列表")
    entities: List[str] = Field(description="实体列表")


class QuestionExtraction(BaseModel):
    """问题抽取结果"""

    questions: List[str] = Field(description="可回答的问题列表")


class SummaryGeneration(BaseModel):
    """摘要生成结果"""

    summary: str = Field(description="内容摘要")
    key_points: List[str] = Field(description="关键要点")


class ReasoningExtraction(BaseModel):
    """推理关系抽取结果"""

    relations: List[Dict[str, str]] = Field(description="推理关系列表")
    confidence: float = Field(description="整体置信度")


# === 统计和监控模型 ===


class MemoryStats(BaseModel):
    """记忆统计信息"""

    total_memories: int = Field(description="总记忆数")
    active_memories: int = Field(description="活跃记忆数")
    archived_memories: int = Field(description="归档记忆数")
    forgotten_memories: int = Field(description="已遗忘记忆数")

    avg_importance_score: float = Field(description="平均重要性分数")
    avg_access_count: float = Field(description="平均访问次数")
    total_associations: int = Field(description="总关联数")

    storage_size_mb: float = Field(description="存储大小(MB)")
    index_size_mb: float = Field(description="索引大小(MB)")


class PerformanceMetrics(BaseModel):
    """性能指标"""

    avg_retrieval_time_ms: float = Field(description="平均检索时间(毫秒)")
    avg_indexing_time_ms: float = Field(description="平均索引时间(毫秒)")
    avg_context_generation_time_ms: float = Field(description="平均引子生成时间(毫秒)")

    cache_hit_rate: float = Field(description="缓存命中率")
    memory_usage_mb: float = Field(description="内存使用(MB)")

    last_updated: float = Field(default_factory=time.time, description="最后更新时间")


# === 配置相关模型 ===


class NamespaceConfig(BaseModel):
    """命名空间配置"""

    namespace: str = Field(description="命名空间名称")
    description: str = Field(description="命名空间描述")
    max_memories: int = Field(100000, description="最大记忆数量")
    retention_days: int = Field(365, description="保留天数")
    enable_cross_query: bool = Field(False, description="是否允许跨空间查询")

    created_at: float = Field(default_factory=time.time, description="创建时间")
    last_accessed: float = Field(default_factory=time.time, description="最后访问时间")
