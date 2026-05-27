# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 模拟引擎 — 训练模拟器 + Omphalos 进化竞技场。"""
from core.simulation.world import SimWorld
from core.simulation.entities import SimEntity
from core.simulation.estimator import TrainingEstimator, DatasetAnalyzer, estimate_training, recommend_params

__all__ = ["SimWorld", "SimEntity", "TrainingEstimator", "DatasetAnalyzer", "estimate_training", "recommend_params"]
