"""
EverMind Configuration Management

渐进式记忆系统的核心配置，支持从简单向量检索到复杂认知功能的平滑升级。
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class RRWeights(BaseModel):
    """
    RR权重配置：面向用户查询的核心权重
    - Relevance: 语义相关性权重
    - Recency: 时间相关性权重
    """

    relevance: float = Field(1.0, ge=0.0, description="语义相关性权重")
    recency: float = Field(0.3, ge=0.0, description="时间相关性权重")


class IFConfig(BaseModel):
    """
    IF配置：系统内部管理参考
    - Importance: 存储策略参考
    - Frequency: 缓存策略参考
    """

    importance_threshold_archive: float = Field(
        1.0, description="重要性低于此值考虑归档"
    )
    frequency_threshold_cache: int = Field(5, description="访问次数高于此值考虑缓存")
    recency_decay_rate: float = Field(0.01, description="时间衰减率，值越大遗忘越快")


class ForgettingConfig(BaseModel):
    """
    自适应遗忘配置
    """

    enable_adaptive_forgetting: bool = Field(False, description="是否启用自适应遗忘")
    time_weight: float = Field(0.4, description="时间因子权重")
    importance_weight: float = Field(0.3, description="重要性因子权重")
    frequency_weight: float = Field(0.2, description="频次因子权重")
    association_weight: float = Field(0.1, description="关联密度因子权重")
    forgetting_threshold: float = Field(0.8, description="遗忘概率阈值")
    observation_period_days: int = Field(30, description="遗忘观察期天数")


class IndexingConfig(BaseModel):
    """
    多粒度索引配置
    """

    enable_concept_extraction: bool = Field(True, description="是否启用概念抽取")
    enable_question_extraction: bool = Field(True, description="是否启用问题抽取")
    enable_summary_generation: bool = Field(True, description="是否启用摘要生成")

    concept_extraction_threshold: float = Field(2.0, description="概念抽取的重要性阈值")
    question_extraction_threshold: float = Field(
        2.5, description="问题抽取的重要性阈值"
    )
    max_concepts_per_memory: int = Field(5, description="每条记忆最大概念数")
    max_questions_per_memory: int = Field(3, description="每条记忆最大问题数")


class ReasoningConfig(BaseModel):
    """
    推理链配置
    """

    enable_reasoning_chains: bool = Field(False, description="是否启用推理链追踪")
    max_chain_depth: int = Field(3, description="最大推理链深度")
    chain_confidence_threshold: float = Field(0.7, description="推理链置信度阈值")


class PerformanceConfig(BaseModel):
    """
    性能相关配置
    """

    context_generation_timeout_ms: int = Field(200, description="引子生成超时(毫秒)")
    retrieval_timeout_ms: int = Field(500, description="检索超时(毫秒)")
    max_memories_per_user: int = Field(100000, description="单用户最大记忆数")
    max_context_memories: int = Field(10, description="单次上下文最大记忆数")

    # Qdrant相关
    qdrant_collection_config: Dict = Field(
        default_factory=lambda: {
            "vectors": {"size": 1024, "distance": "Cosine"},
            "optimizers_config": {"memmap_threshold": 20000},
            "hnsw_config": {"m": 16, "ef_construct": 100},
        },
        description="Qdrant集合配置",
    )


class EverMindConfig(BaseModel):
    """
    EverMind主配置类：渐进式功能开关设计
    """

    # === 核心功能开关 ===
    enable_multi_granular_indexing: bool = Field(True, description="是否启用多粒度索引")
    enable_association_tracking: bool = Field(True, description="是否启用关联追踪")
    enable_context_injection: bool = Field(True, description="是否启用引子机制")

    # === 高级功能开关 ===
    enable_reasoning_chains: bool = Field(False, description="是否启用推理链")
    enable_adaptive_forgetting: bool = Field(False, description="是否启用自适应遗忘")

    # === 分片配置 ===
    default_namespace: str = Field("default", description="默认命名空间")
    enable_cross_namespace_query: bool = Field(
        False, description="是否允许跨命名空间查询"
    )

    # === 权重配置 ===
    rr_weights: RRWeights = Field(default_factory=RRWeights, description="RR权重配置")
    weights_by_task: Dict[str, RRWeights] = Field(
        default_factory=lambda: {
            "default": RRWeights(),
            "factual_qa": RRWeights(relevance=1.5, recency=0.2),
            "conversation": RRWeights(relevance=0.8, recency=0.6),
            "search": RRWeights(relevance=1.8, recency=0.1),
        },
        description="按任务类型的权重配置",
    )

    # === 子配置 ===
    if_config: IFConfig = Field(default_factory=IFConfig, description="IF配置")
    forgetting_config: ForgettingConfig = Field(
        default_factory=ForgettingConfig, description="遗忘配置"
    )
    indexing_config: IndexingConfig = Field(
        default_factory=IndexingConfig, description="索引配置"
    )
    reasoning_config: ReasoningConfig = Field(
        default_factory=ReasoningConfig, description="推理配置"
    )
    performance_config: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="性能配置"
    )

    class Config:
        validate_assignment = True

    def get_task_weights(self, task_type: str) -> RRWeights:
        """获取指定任务类型的权重配置"""
        return self.weights_by_task.get(task_type, self.rr_weights)

    def is_feature_enabled(self, feature: str) -> bool:
        """检查功能是否启用"""
        feature_map = {
            "indexing": self.enable_multi_granular_indexing,
            "association": self.enable_association_tracking,
            "context": self.enable_context_injection,
            "reasoning": self.enable_reasoning_chains,
            "forgetting": self.enable_adaptive_forgetting,
        }
        return feature_map.get(feature, False)
