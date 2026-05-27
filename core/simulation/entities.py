# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 模拟实体 — 数字生命体。"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SimEntity:
    name: str
    faction: str = ""
    generation: int = 1

    # 六维属性
    reasoning: float = 70.0
    creativity: float = 70.0
    social: float = 70.0
    adaptability: float = 70.0
    knowledge: float = 70.0
    combat: float = 70.0

    # 状态
    health: float = 100.0
    energy: float = 100.0
    level: int = 1
    exp: float = 0.0
    position: tuple = (0, 0)
    alive: bool = True

    # 记录
    story: str = ""
    allies: List[str] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.faction:
            self.faction = random.choice(["探索者", "创造者", "守护者", "挑战者", "观察者"])
        if not self.story:
            self.story = f"G{self.generation} 诞生。"

    def evolve_attr(self, attr: str, amount: float):
        if hasattr(self, attr):
            setattr(self, attr, min(150, max(0, getattr(self, attr) + amount)))

    def take_damage(self, amount: float, reason: str = ""):
        self.health = max(0, self.health - amount)
        if self.health <= 0:
            self.alive = False

    def gain_exp(self, amount: float):
        self.exp += amount
        while self.exp >= self.level * 100:
            self.exp -= self.level * 100
            self.level += 1
            boost = random.choice(["reasoning", "creativity", "social",
                                   "adaptability", "knowledge", "combat"])
            self.evolve_attr(boost, random.uniform(1, 5))

    def get_score(self) -> float:
        return (self.reasoning + self.creativity + self.social +
                self.adaptability + self.knowledge + self.combat) / 6 + self.level * 3

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "faction": self.faction, "generation": self.generation,
            "level": self.level, "score": round(self.get_score(), 1),
            "alive": self.alive, "health": round(self.health, 1),
            "attrs": {
                "推理": round(self.reasoning, 1), "创意": round(self.creativity, 1),
                "社交": round(self.social, 1), "适应": round(self.adaptability, 1),
                "知识": round(self.knowledge, 1), "战斗": round(self.combat, 1),
            },
        }

    @classmethod
    def create_random(cls, name: str, faction: str = "", gen: int = 1) -> "SimEntity":
        return cls(
            name=name, faction=faction, generation=gen,
            reasoning=random.uniform(50, 100), creativity=random.uniform(50, 100),
            social=random.uniform(50, 100), adaptability=random.uniform(50, 100),
            knowledge=random.uniform(50, 100), combat=random.uniform(50, 100),
            position=(random.randint(0, 79), random.randint(0, 59)),
        )
