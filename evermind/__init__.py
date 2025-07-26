"""
Evermind: 为智能体打造的“神级记忆装备”，一个可实现永久存活与自我进化的认知核心。
"""

from .config import MnemonConfig, RRIFWeights, MaintenanceConfig
from .manager import MemoryManager
from .models import (
    MemoryRecord,
    MemoryMetadata,
    MemoryStatus,
    QueryResult,
    RetrievedMemory,
)
from .protocols import ITaskQueue
from .task_queues.in_process import InProcessTaskQueue

# 使用 __all__ 来明确定义包的公共 API
# 当用户执行 `from mnemon import *` 时，只有这些名称会被导入。
__all__ = [
    "MemoryManager",
    "MnemonConfig",
    "RRIFWeights",
    "MaintenanceConfig",
    "MemoryRecord",
    "MemoryMetadata",
    "MemoryStatus",
    "QueryResult",
    "RetrievedMemory",
    "ITaskQueue",
    "InProcessTaskQueue",
]
