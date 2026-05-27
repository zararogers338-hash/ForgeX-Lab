# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

# core/task_queue.py - ForgeX v2 異步任務引擎（替代阻塞式訓練）
import threading
import traceback
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any
from datetime import datetime

from core.logger import log
from core.utils import get_timestamp


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str = ""
    created_at: str = field(default_factory=get_timestamp)
    started_at: str = ""
    finished_at: str = ""
    logs: List[str] = field(default_factory=list)
    _cancel_flag: bool = False

    def cancel(self):
        self._cancel_flag = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag

    def update_progress(self, progress: float, message: str = ""):
        self.progress = min(progress, 100.0)
        if message:
            self.message = message
            self.logs.append(f"[{get_timestamp()}] {message}")
            # 防止日志无限增长导致 OOM
            if len(self.logs) > 2000:
                self.logs = self.logs[-1500:]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_count": len(self.logs),
        }


class TaskQueue:
    """簡單的後台任務隊列（單線程執行，避免 GPU 搶佔）"""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._queue: List[str] = []
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    def submit(self, name: str, func: Callable, *args, **kwargs) -> str:
        """提交任務，返回任務 ID"""
        task_id = f"task_{get_timestamp()}_{len(self._tasks)}"
        task = Task(id=task_id, name=name, func=func, args=args, kwargs=kwargs)

        with self._lock:
            self._tasks[task_id] = task
            self._queue.append(task_id)

        log(f"任務已提交: {name} ({task_id})")
        self._ensure_worker()
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[dict]:
        return [t.to_dict() for t in reversed(list(self._tasks.values()))]

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            task.cancel()
            task.status = TaskStatus.CANCELLED
            task.finished_at = get_timestamp()
            log(f"任務已取消: {task.name}")
            return True
        return False

    def get_task_logs(self, task_id: str) -> List[str]:
        task = self._tasks.get(task_id)
        return task.logs if task else []

    def _ensure_worker(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self):
        idle_count = 0
        while True:
            task_id = None
            with self._lock:
                # 找下一個 pending 任務
                for tid in self._queue:
                    if self._tasks[tid].status == TaskStatus.PENDING:
                        task_id = tid
                        break

            if task_id is None:
                idle_count += 1
                time.sleep(0.5)
                # 如果連續空閒超過 10 秒且沒有任何待處理任務，退出
                if idle_count > 20:
                    with self._lock:
                        pending = any(t.status == TaskStatus.PENDING for t in self._tasks.values())
                    if not pending:
                        break
                    idle_count = 0
                continue

            idle_count = 0

            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = get_timestamp()
            log(f"開始執行任務: {task.name}")

            try:
                # 注入 task 對象讓函數可以更新進度
                result = task.func(*task.args, task=task, **task.kwargs)
                if not task.is_cancelled:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.progress = 100.0
                    task.message = "完成"
                    log(f"✅ 任務完成: {task.name}")
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.message = f"錯誤: {e}"
                log(f"❌ 任務失敗: {task.name} - {e}")
                task.logs.append(traceback.format_exc())
            finally:
                task.finished_at = get_timestamp()
                with self._lock:
                    if task_id in self._queue:
                        self._queue.remove(task_id)


# 全局任務隊列
task_queue = TaskQueue()
