# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 事件总线 — 模块间零耦合通信。

用法:
    EventBus.on("train_complete", my_handler)
    EventBus.emit("train_complete", lora_path="/path/to/lora")
"""
import threading
from typing import Dict, List, Callable

class EventBus:
    _listeners: Dict[str, List[Callable]] = {}
    _lock = threading.Lock()

    @classmethod
    def on(cls, event: str, callback: Callable):
        with cls._lock:
            cls._listeners.setdefault(event, []).append(callback)

    @classmethod
    def off(cls, event: str, callback: Callable = None):
        with cls._lock:
            if callback is None:
                cls._listeners.pop(event, None)
            elif event in cls._listeners:
                cls._listeners[event] = [cb for cb in cls._listeners[event] if cb != callback]

    @classmethod
    def emit(cls, event: str, **data):
        with cls._lock:
            listeners = list(cls._listeners.get(event, []))
        for cb in listeners:
            try:
                cb(**data)
            except Exception:
                pass

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._listeners.clear()


class Events:
    # 训练
    TRAIN_START    = "train_start"
    TRAIN_PROGRESS = "train_progress"
    TRAIN_COMPLETE = "train_complete"
    TRAIN_ERROR    = "train_error"
    # 锻造
    FORGE_START    = "forge_start"
    FORGE_PROGRESS = "forge_progress"
    FORGE_COMPLETE = "forge_complete"
    FORGE_ERROR    = "forge_error"
    # 反馈
    FEEDBACK_ADDED = "feedback_added"
    # 模拟
    SIM_TICK       = "sim_tick"
    SIM_BATTLE     = "sim_battle"
    SIM_EVENT      = "sim_event"
