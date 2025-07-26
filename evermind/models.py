import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# --- Core Data Structures ---
# 定义了我们记忆系统中流转的核心数据模型。


class MemoryStatus(str, Enum):
    """
    定义一个记忆记录的生命周期状态。
    """

    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"


class MemoryMetadata(BaseModel):
    """
    记忆的元数据，包含了所有用于RRIF模型评估和管理的信息。
    """

    source_type: str = Field(
        description="记忆来源，例如：'user_chat', 'api_call', 'self_reflection'"
    )
    interaction_id: Optional[str] = Field(None, description="关联的会话或任务ID")
    importance_score: Optional[float] = Field(
        0.0, description="[0.0, 4.0]，由LLM评估的重要性分数"
    )
    access_count: int = Field(0, description="记忆被访问的次数，用于计算Frequency")
    status: MemoryStatus = Field(MemoryStatus.ACTIVE, description="记忆的当前状态")
    is_compressed: bool = Field(False, description="内容字段是否被物理压缩")
    archive_location: Optional[str] = Field(None, description="在冷存储中的路径或URI")
    custom_data: Dict[str, Any] = Field(
        default_factory=dict, description="用户可自定义的额外数据"
    )
    questions: List[str] = Field(
        default_factory=list, description="从内容中抽取出的、可被回答的问题"
    )


class MemoryRecord(BaseModel):
    """
    表示一条完整的、可存储的记忆记录。
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="记忆的唯一标识符"
    )
    content: Union[str, bytes] = Field(
        description="记忆的原始文本内容，或被压缩后的字节"
    )
    content_embedding: Optional[List[float]] = Field(
        None, description="内容本身的向量表示"
    )
    timestamp: float = Field(
        default_factory=time.time, description="记忆创建的Unix时间戳"
    )
    metadata: MemoryMetadata = Field(
        default_factory=MemoryMetadata, description="记忆的元数据"
    )


class RetrievedMemory(BaseModel):
    """
    在查询过程中，被检索出的记忆及其相关信息。
    """

    id: str
    content: str
    metadata: MemoryMetadata
    score: float = Field(description="经过RRIF模型计算出的最终相关性得分")
    type: str = Field(description="记忆类型，例如：'episodic', 'semantic'")


class QueryResult(BaseModel):
    """
    对`query`接口调用的最终返回结果。
    """

    synthesized_answer: Optional[str] = Field(
        None, description="由LLM基于检索到的记忆合成的最终答案。如果未请求，则为None。"
    )
    retrieved_memories: List[RetrievedMemory] = Field(
        description="用于生成答案的、经过RRIF排序的溯源记忆列表"
    )


# --- LLM Structured Output Models ---
# 定义了用于解析LLM结构化输出的Pydantic模型。


class ImportanceRating(BaseModel):
    """用于解析LLM对记忆重要性评分的结构。"""

    score: float = Field(description="[0.0, 4.0] 范围内的重要性分数")
    reason: str = Field(description="给出该分数的简要理由")


class QuestionExtraction(BaseModel):
    """用于解析LLM从文本中抽取出的问题的结构。"""

    questions: List[str] = Field(description="从文本中抽取的、可被回答的问题列表")


class QueryPlan(BaseModel):
    """用于解析LLM对用户查询的规划结果。"""

    vector_search_query: str = Field(description="优化后的、用于向量搜索的查询语句")
    requires_knowledge_graph: bool = Field(
        description="判断本次查询是否需要查询知识图谱"
    )
    graph_query: Optional[str] = Field(
        None, description="如果需要，生成的可在图数据库中执行的查询语句 (e.g., Cypher)"
    )


class ReflectionResult(BaseModel):
    """用于解析LLM对记忆簇进行反思的结果。"""

    insights: List[str] = Field(description="从记忆中提炼出的核心洞见或模式。")
    new_knowledge_triplets: List[List[str]] = Field(
        description="抽出的新知识三元组 (主体, 关系, 客体)。"
    )


class FusedKnowledge(BaseModel):
    """用于解析LLM对知识进行融合和归一的结果。"""

    fused_triplets: List[List[str]] = Field(
        description="经过融合与归一的、标准化的知识三元组。"
    )
