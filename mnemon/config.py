from typing import Dict

from pydantic import BaseModel, Field

# --- Configuration Models ---
# 定义了MNEMON SDK的所有可配置项。
# 用户在实例化MemoryManager时传入一个MnemonConfig对象来定制系统行为。


class RRIFWeights(BaseModel):
    """
    定义RRIF模型中各个维度的权重。
    权重可以根据不同的任务类型（task_type）进行动态调整。
    """

    relevance: float = 1.0
    recency: float = 0.5
    importance: float = 1.5
    frequency: float = 0.8


class MaintenanceConfig(BaseModel):
    """
    配置所有与记忆维护相关的参数。
    """

    # --- Feature Flags for Maintenance ---
    enable_consolidation: bool = Field(
        True, description="是否启用记忆固化（如合并相似记忆）"
    )
    enable_archiving: bool = Field(
        True, description="是否启用记忆归档（将冷数据迁移到低成本存储）"
    )
    enable_hard_deletion: bool = Field(
        False, description="【危险】是否启用物理删除。默认关闭以防数据丢失。"
    )

    # --- Thresholds for Maintenance ---
    archive_age_threshold_days: int = Field(
        365, description="记忆超过多少天后可被视为归档候选"
    )
    hard_delete_age_threshold_days: int = Field(
        90, description="无价值记忆超过多少天后可被视为删除候选"
    )
    storage_usage_threshold_for_archiving: float = Field(
        0.9, description="主数据库使用率超过此阈值（如90%）时，可触发归档"
    )


class MnemonConfig(BaseModel):
    """
    MNEMON SDK 的主配置类。
    """

    # --- Core Feature Flags ---
    enable_semantic_memory: bool = Field(
        True, description="是否启用语义记忆（知识图谱）"
    )
    enable_meta_reflection: bool = Field(
        True, description="是否启用元认知反思（从经验中学习新知识）"
    )

    # --- RRIF Model Parameters ---
    recency_decay_rate: float = Field(
        0.01, description="时效性分数的衰减率。值越大，记忆“过时”得越快。"
    )
    importance_threshold_for_ingestion: float = Field(
        0.0,
        description="重要性分数低于此阈值的记忆将不会被深度处理。注意：根据我们的最新设计，所有记忆都会被初步记录，此阈值可能用于决定处理优先级而非直接丢弃。",
    )
    importance_threshold_for_question_extraction: float = Field(
        3.5, description="重要性分数高于此阈值的记忆，将触发额外的问题抽取流程"
    )

    # --- RRIF Weights Configuration ---
    weights_by_task: Dict[str, RRIFWeights] = Field(
        default_factory=lambda: {
            "default": RRIFWeights(),
            "factual_qa": RRIFWeights(relevance=1.8, importance=2.0, recency=0.2),
            "casual_chat": RRIFWeights(relevance=0.8, recency=1.5, importance=0.5),
        },
        description="为不同任务类型预设的RRIF权重，允许动态调整检索策略",
    )

    # --- Maintenance Configuration ---
    maintenance: MaintenanceConfig = Field(
        default_factory=MaintenanceConfig, description="记忆维护相关的配置"
    )

    class Config:
        # Pydantic v2 的配置方式
        validate_assignment = True
