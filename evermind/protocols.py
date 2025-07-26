from abc import ABC, abstractmethod
from typing import Callable

# --- Interface Protocols ---
# 定义了系统依赖的外部服务的抽象接口。
# 这允许用户插入任何符合这些协议的自定义实现。


class ITaskQueue(ABC):
    """
    异步任务队列的协议。
    MNEMON使用它来处理耗时的操作（如LLM调用、记忆维护），避免阻塞主流程。
    """

    @abstractmethod
    def submit_task(self, function: Callable, *args, **kwargs) -> str:
        """
        提交一个任务到队列中执行。

        Args:
            function: 需要在后台执行的函数。
            *args: 函数的位置参数。
            **kwargs: 函数的关键字参数。

        Returns:
            一个代表此任务的唯一ID。
        """
        pass
