# -*- coding: utf-8 -*-
# Decompiled source for open-source release. Original user package used a lightweight packaging obfuscation wrapper.

"""ForgeX 模拟世界 — 实体在网格世界中交互，自动生成训练数据。

训练数据生成:
  - 战斗决策: "A(推理90) vs B(战斗85)，A选择策略X → 结果Y"
  - 合作场景: "A和B合作，各取所长完成任务"
  - 生存决策: "低血量时选择逃跑/战斗/求助"
  - 进化选择: "哪些属性组合更适合生存"

这些数据可以用来训练模型的决策能力和推理能力。
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core import DATASETS_DIR, log
from core.simulation.entities import SimEntity


FACTIONS = ["探索者", "创造者", "守护者", "挑战者", "观察者"]
TERRAIN_TYPES = ["平原", "森林", "山脉", "沙漠", "遗迹", "水晶矿"]
TERRAIN_MODIFIERS = {
    "平原": {"social": 3, "adaptability": 2},
    "森林": {"creativity": 4, "adaptability": 3},
    "山脉": {"reasoning": 4, "combat": 2},
    "沙漠": {"adaptability": 5, "social": -2},
    "遗迹": {"knowledge": 8, "reasoning": 3},
    "水晶矿": {"creativity": 5, "knowledge": 3},
}

EVENT_TEMPLATES = [
    ("天灾", "{e}遭遇能量风暴", {"health": -15, "adaptability": 3}),
    ("发现", "{e}发现了远古知识", {"knowledge": 8, "exp": 30}),
    ("顿悟", "{e}获得思维突破", {"reasoning": 10, "exp": 25}),
    ("变异", "{e}属性突变增强", {"random_boost": 12}),
    ("瘟疫", "{e}感染了未知疾病", {"health": -20}),
    ("机遇", "{e}遇到了稀有资源", {"exp": 50}),
]


class SimWorld:
    """模拟世界引擎"""

    def __init__(self, width: int = 80, height: int = 60, seed: int = None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 99999)
        self._rng = random.Random(self.seed)

        # 状态
        self.entities: List[SimEntity] = []
        self.round_num: int = 0
        self.training_data: List[Dict] = []

        # 统计
        self.stat_battles = 0
        self.stat_coops = 0
        self.stat_births = 0
        self.events_log: List[str] = []

        # 地形
        self.terrain = self._generate_terrain()

        # 运行控制
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._speed = 2.0

    def _generate_terrain(self) -> List[List[str]]:
        """简化版地形生成"""
        grid = [["平原"] * self.width for _ in range(self.height)]
        # 随机放置地形块
        for terrain_type in TERRAIN_TYPES:
            for _ in range(self._rng.randint(3, 8)):
                cx = self._rng.randint(5, self.width - 6)
                cy = self._rng.randint(5, self.height - 6)
                r = self._rng.randint(3, 7)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx*dx + dy*dy <= r*r:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                grid[ny][nx] = terrain_type
        return grid

    def get_terrain(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.terrain[y][x]
        return "平原"

    # ── 实体管理 ──
    def spawn_random(self, count: int = 20):
        names = ["星火", "幽影", "晓光", "深渊", "灵风", "铁心", "幻梦", "苍穹",
                 "烈焰", "静水", "雷霆", "月华", "霜刃", "紫电", "碧落", "玄武"]
        for i in range(count):
            name = self._rng.choice(names) + str(self._rng.randint(1, 999))
            faction = self._rng.choice(FACTIONS)
            e = SimEntity.create_random(name, faction)
            e.position = (self._rng.randint(0, self.width - 1),
                         self._rng.randint(0, self.height - 1))
            self.entities.append(e)

    def get_alive(self) -> List[SimEntity]:
        return [e for e in self.entities if e.alive]

    # ── 模拟循环 ──
    def start(self, tick_cb: Callable = None):
        if self._running:
            return
        self._running = True
        def _loop():
            while self._running:
                self.tick()
                if tick_cb:
                    tick_cb(self.round_num)
                time.sleep(max(0.05, 1.0 / self._speed))
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def set_speed(self, speed: float):
        self._speed = max(0.1, min(20.0, speed))

    @property
    def is_running(self) -> bool:
        return self._running

    def tick(self):
        """执行一轮"""
        self.round_num += 1
        alive = self.get_alive()
        if not alive:
            self._running = False
            return

        # 1. 移动
        for e in alive:
            self._move(e)

        # 2. 地形效果
        for e in alive:
            terrain = self.get_terrain(*e.position)
            mods = TERRAIN_MODIFIERS.get(terrain, {})
            for attr, val in mods.items():
                e.evolve_attr(attr, val * 0.01)

        # 3. 交互
        self._process_interactions(alive)

        # 4. 随机事件
        if self._rng.random() < 0.2:
            self._random_event(alive)

        # 5. 恢复 & 消耗
        for e in alive:
            e.health = min(100, e.health + 0.5)
            e.energy = min(100, e.energy + 1.0) - 0.3
            if e.energy <= 0:
                e.take_damage(5, "体力耗尽")

        # 6. 繁殖
        if self.round_num % 20 == 0:
            self._breeding(alive)

        # 7. 导出训练数据
        if self.round_num % 100 == 0 and self.training_data:
            self._export()

    def _move(self, e: SimEntity):
        x, y = e.position
        dx = self._rng.randint(-2, 2)
        dy = self._rng.randint(-2, 2)
        e.position = (max(0, min(self.width - 1, x + dx)),
                     max(0, min(self.height - 1, y + dy)))

    def _process_interactions(self, alive: List[SimEntity]):
        # 按区域分组
        cells: Dict = {}
        for e in alive:
            key = (e.position[0] // 5, e.position[1] // 5)
            cells.setdefault(key, []).append(e)

        for group in cells.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    e1, e2 = group[i], group[j]
                    if not e1.alive or not e2.alive:
                        continue
                    if e1.faction == e2.faction:
                        self._cooperate(e1, e2)
                    elif self._rng.random() < 0.3:
                        self._battle(e1, e2)

    def _battle(self, e1: SimEntity, e2: SimEntity):
        # 战斗力 = 综合属性 + 随机波动
        p1 = (e1.reasoning * 0.3 + e1.combat * 0.35 + e1.adaptability * 0.2 +
              e1.creativity * 0.15 + e1.level * 5) * self._rng.uniform(0.8, 1.2)
        p2 = (e2.reasoning * 0.3 + e2.combat * 0.35 + e2.adaptability * 0.2 +
              e2.creativity * 0.15 + e2.level * 5) * self._rng.uniform(0.8, 1.2)

        winner, loser = (e1, e2) if p1 > p2 else (e2, e1)
        damage = abs(p1 - p2) * 0.3 + self._rng.uniform(5, 15)
        loser.take_damage(damage)
        winner.gain_exp(20 + loser.level * 5)
        winner.evolve_attr("combat", 1)
        self.stat_battles += 1

        # 生成训练数据 — 战斗决策
        terrain = self.get_terrain(*winner.position)
        self.training_data.append({
            "instruction": (
                f"在{terrain}地形，{winner.name}(Lv{winner.level}, "
                f"推理{winner.reasoning:.0f}, 战斗{winner.combat:.0f}, "
                f"适应{winner.adaptability:.0f}) "
                f"遇到了敌对阵营的{loser.name}(Lv{loser.level}, "
                f"战斗{loser.combat:.0f})。应该如何决策？"
            ),
            "output": (
                f"分析: {winner.name}的综合战斗力({p1:.0f})优于对手({p2:.0f})。"
                f"推理属性({winner.reasoning:.0f})提供了战术优势，"
                f"适应力({winner.adaptability:.0f})帮助利用{terrain}地形。"
                f"决策: 主动出击。结果: 造成{damage:.0f}伤害，获胜。"
                + (f" {loser.name}被淘汰。" if not loser.alive else "")
            ),
        })

    def _cooperate(self, e1: SimEntity, e2: SimEntity):
        if e1.name not in e2.allies:
            e2.allies.append(e1.name)
        if e2.name not in e1.allies:
            e1.allies.append(e2.name)
        e1.evolve_attr("social", 1)
        e2.evolve_attr("social", 1)
        e1.gain_exp(5)
        e2.gain_exp(5)
        self.stat_coops += 1

    def _breeding(self, alive: List[SimEntity]):
        if len(alive) >= 100:
            return
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                e1, e2 = alive[i], alive[j]
                if (e1.faction == e2.faction and
                    e1.name in e2.allies and self._rng.random() < 0.1):
                    child = SimEntity(
                        name=f"{e1.name[:3]}{e2.name[:3]}_{self._rng.randint(1,99)}",
                        faction=self._rng.choice([e1.faction, e2.faction]),
                        generation=max(e1.generation, e2.generation) + 1,
                    )
                    attrs = ["reasoning", "creativity", "social",
                             "adaptability", "knowledge", "combat"]
                    for attr in attrs:
                        v = (getattr(e1, attr) + getattr(e2, attr)) / 2
                        v += self._rng.gauss(0, 5)
                        setattr(child, attr, max(30, min(150, v)))
                    child.position = e1.position
                    self.entities.append(child)
                    self.stat_births += 1

                    # 训练数据 — 遗传进化
                    self.training_data.append({
                        "instruction": (
                            f"{e1.name}(推理{e1.reasoning:.0f}, 创意{e1.creativity:.0f}) "
                            f"和{e2.name}(知识{e2.knowledge:.0f}, 社交{e2.social:.0f}) "
                            f"的后代会有什么特点？"
                        ),
                        "output": (
                            f"后代{child.name}继承了双亲的优势: "
                            f"推理{child.reasoning:.0f}, 创意{child.creativity:.0f}, "
                            f"知识{child.knowledge:.0f}, 社交{child.social:.0f}。"
                            f"属于第{child.generation}代，综合评分{child.get_score():.1f}。"
                        ),
                    })
                    return  # 每轮最多繁殖 1 个

    def _random_event(self, alive: List[SimEntity]):
        if not alive:
            return
        tpl = self._rng.choice(EVENT_TEMPLATES)
        event_type, desc_tpl, effects = tpl
        target = self._rng.choice(alive)
        desc = desc_tpl.format(e=target.name)

        for key, val in effects.items():
            if key == "health":
                target.health = max(0, target.health + val)
            elif key == "exp":
                target.gain_exp(val)
            elif key == "random_boost":
                attr = self._rng.choice(["reasoning", "creativity", "social",
                                         "adaptability", "knowledge", "combat"])
                target.evolve_attr(attr, val)
            elif hasattr(target, key):
                target.evolve_attr(key, val)

        self.events_log.append(f"R{self.round_num}: {desc}")

    def _export(self):
        """导出训练数据"""
        if not self.training_data:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = Path(DATASETS_DIR) / f"sim_r{self.round_num}_{ts}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for item in self.training_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"📊 模拟导出 {len(self.training_data)} 条训练数据 → {out.name}")
        self.training_data.clear()

    def export_all(self) -> Optional[Path]:
        """手动导出当前所有训练数据"""
        if not self.training_data:
            return None
        self._export()
        return Path(DATASETS_DIR)

    def get_stats(self) -> Dict:
        alive = self.get_alive()
        factions = {}
        for e in alive:
            factions[e.faction] = factions.get(e.faction, 0) + 1
        return {
            "round": self.round_num, "alive": len(alive),
            "total": len(self.entities), "dead": len(self.entities) - len(alive),
            "factions": factions, "battles": self.stat_battles,
            "coops": self.stat_coops, "births": self.stat_births,
            "data_pending": len(self.training_data),
            "avg_score": round(sum(e.get_score() for e in alive) / max(len(alive), 1), 1),
            "top": max(alive, key=lambda e: e.get_score()).name if alive else "-",
        }

    def get_leaderboard(self, top_n: int = 10) -> List[Dict]:
        alive = sorted(self.get_alive(), key=lambda e: e.get_score(), reverse=True)
        return [e.to_dict() for e in alive[:top_n]]
