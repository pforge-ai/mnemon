import logging
from typing import Callable

from ..protocols import ITaskQueue

logger = logging.getLogger(__name__)


class InProcessTaskQueue(ITaskQueue):
    """
    一个在当前进程中同步执行任务的简单任务队列实现。

    这个类主要用于开发、测试或不需要后台处理的简单场景。
    它会直接执行传入的函数，而不是将其分派到后台worker。
    """

    def submit_task(self, function: Callable, *args, **kwargs) -> str:
        """
                同步执行一个任务。
        ————
                Args:
                    function: 需要执行的函数。
                    *args: 函数的位置参数。
                    **kwargs: 函数的关键字参数。

                Returns:
                    一个表示“任务已完成”的简单字符串。
        """
        task_name = function.__name__
        logger.debug(f"Executing task '{task_name}' in-process (synchronously).")
        try:
            function(*args, **kwargs)
            logger.debug(f"Task '{task_name}' executed successfully.")
            return f"completed_in_process_{task_name}"
        except Exception as e:
            logger.error(
                f"An error occurred during in-process task execution for '{task_name}': {e}",
                exc_info=True,
            )
            # 在同步执行中，我们可以选择重新抛出异常，让调用者知道任务失败了。
            raise
