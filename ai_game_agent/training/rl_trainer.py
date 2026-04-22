"""
RL Trainer
==========
Reinforcement learning loop for the AI Game Dev agent.

Each episode:
  1. GENERATE — agent produces pixel art / code / design
  2. EVALUATE — GameEvaluator scores the output (0–100)
  3. REWARD / PENALISE — result stored in ExperienceMemory
  4. IMPROVE — next generation uses few_shot_prompt from memory

The agent improves over time by:
  - Seeing its own best examples as positive demonstrations
  - Seeing its own worst examples as "do NOT do this"
  - Gradually increasing the quality threshold to unlock harder tasks

Curriculum:
  Level 1 (score < 40): Basic — single tile, simple character, 20-line script
  Level 2 (score 40–65): Medium — animated character, multi-tile map, full scene
  Level 3 (score 65–80): Hard — full game scene, enemies, items, save/load
  Level 4 (score > 80): Expert — complete playable game, polished art, cutscenes
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Callable

from .game_evaluator import GameEvaluator, EvalResult
from .experience_memory import ExperienceMemory


# ── Training config ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    max_episodes: int = 100
    target_score: float = 75.0       # stop when rolling avg hits this
    penalty_threshold: float = 40.0  # below this = penalised episode
    reward_threshold: float = 65.0   # above this = rewarded episode
    window_size: int = 10            # rolling average window
    action_types: list = None        # None = all types

    def __post_init__(self):
        if self.action_types is None:
            self.action_types = [
                "draw_character",
                "draw_tile",
                "draw_prop",
                "generate_scene_code",
                "generate_player_code",
                "generate_enemy_code",
                "design_map",
                "design_game",
            ]


# ── Curriculum ────────────────────────────────────────────────────────────────

CURRICULUM = [
    {
        "level": 1,
        "name": "Basics",
        "min_score": 0,
        "tasks": [
            {
                "action": "draw_character",
                "prompt": "Draw a simple warrior character facing south. Use DB32 palette. Keep it clean with ≤16 colours.",
                "params": {"archetype": "warrior", "direction": "south", "size": 32},
            },
            {
                "action": "draw_tile",
                "prompt": "Draw a grass tile, 16×16 pixels, clean palette, no noise.",
                "params": {"tile_type": "grass", "size": 16},
            },
            {
                "action": "generate_player_code",
                "prompt": "Write minimal GDScript for a CharacterBody2D player with WASD movement and a health variable.",
                "params": {"complexity": "simple"},
            },
        ],
    },
    {
        "level": 2,
        "name": "Intermediate",
        "min_score": 40,
        "tasks": [
            {
                "action": "draw_character",
                "prompt": "Draw a detailed mage character with a staff, 4 directions, DB32 palette, clean edges.",
                "params": {"archetype": "mage", "all_directions": True, "size": 32},
            },
            {
                "action": "generate_scene_code",
                "prompt": "Write GDScript for a game scene with a player, a tilemap, enemy spawner, and score label.",
                "params": {"complexity": "medium"},
            },
            {
                "action": "design_map",
                "prompt": "Design a village map with grass biome, water border, forest patches, and a path.",
                "params": {"biomes": ["grass", "water", "forest"], "size": "medium"},
            },
        ],
    },
    {
        "level": 3,
        "name": "Advanced",
        "min_score": 65,
        "tasks": [
            {
                "action": "generate_enemy_code",
                "prompt": "Write GDScript for an enemy with patrol AI, attack range detection, health, and death animation trigger.",
                "params": {"complexity": "hard"},
            },
            {
                "action": "design_game",
                "prompt": "Design a complete isekai RPG: player stats, inventory, quest system, NPC dialog, and save/load.",
                "params": {"genre": "isekai_rpg", "complexity": "full"},
            },
        ],
    },
    {
        "level": 4,
        "name": "Expert",
        "min_score": 80,
        "tasks": [
            {
                "action": "design_game",
                "prompt": (
                    "Create a complete, polished, playable isekai village sim RPG in the style of Vagabond's art quality. "
                    "Include: animated sprites for all characters, procedural world map, combat system, crafting, "
                    "NPC AI with schedules, quest journal, and atmospheric music triggers."
                ),
                "params": {"genre": "isekai_rpg", "complexity": "expert"},
            },
        ],
    },
]


# ── RL Trainer ────────────────────────────────────────────────────────────────

class RLTrainer:
    """
    Main training orchestrator.
    Runs episodes, scores them, stores results, raises curriculum difficulty.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        on_episode: Optional[Callable] = None,
    ):
        self.config = config or TrainingConfig()
        self.memory = ExperienceMemory()
        self.evaluator = GameEvaluator()
        self.on_episode = on_episode  # callback(episode_result_dict) for streaming to UI

        self._running = False
        self._episode_count = 0
        self._score_history: list[float] = []
        self._current_level = 1
        self._generate_fn: Optional[Callable] = None

    def set_generator(self, fn: Callable):
        """Inject the generation function (calls agent endpoints)."""
        self._generate_fn = fn

    @property
    def rolling_avg(self) -> float:
        if not self._score_history:
            return 0.0
        window = self._score_history[-self.config.window_size:]
        return sum(window) / len(window)

    @property
    def current_curriculum_level(self) -> dict:
        for lvl in reversed(CURRICULUM):
            if self.rolling_avg >= lvl["min_score"]:
                return lvl
        return CURRICULUM[0]

    def stop(self):
        self._running = False

    async def run(self) -> dict:
        """
        Run the full training loop.
        Returns summary stats when done.
        """
        self._running = True
        print(f"\n{'='*60}")
        print("🎮 AI Game Dev — RL Training Starting")
        print(f"   Max episodes : {self.config.max_episodes}")
        print(f"   Target score : {self.config.target_score}")
        print(f"   Penalty below: {self.config.penalty_threshold}")
        print(f"{'='*60}\n")

        for ep_num in range(1, self.config.max_episodes + 1):
            if not self._running:
                print("[Trainer] Stopped by user")
                break

            self._episode_count = ep_num
            curr = self.current_curriculum_level
            task = self._pick_task(curr)

            print(f"\n── Episode {ep_num}/{self.config.max_episodes} "
                  f"[Level {curr['level']}: {curr['name']}] ──")
            print(f"   Task: {task['action']}")
            print(f"   Prompt: {task['prompt'][:80]}...")

            # Add memory context to prompt
            memory_ctx = self.memory.few_shot_prompt(task["action"])
            full_prompt = memory_ctx + "\n" + task["prompt"]

            # Generate
            try:
                output = await self._generate(task, full_prompt)
            except Exception as e:
                print(f"   ⚠️  Generation failed: {e}")
                output = {"error": str(e)}

            # Evaluate
            result = self._evaluate(task["action"], output)

            # Store in memory
            ep = self.memory.add(
                action_type=task["action"],
                action_params=task["params"],
                output_summary=output.get("summary", task["action"]),
                pixel_art_score=result.pixel_art_score,
                code_score=result.code_score,
                design_score=result.design_score,
                total_score=result.total_score,
                penalties=result.penalties,
                bonuses=result.bonuses,
            )

            self._score_history.append(result.total_score)

            # Print episode result
            status = "✅ REWARD" if result.total_score >= self.config.reward_threshold else \
                     "❌ PENALTY" if result.total_score < self.config.penalty_threshold else \
                     "⚠️  OK"
            print(f"   {status} — Score: {result.total_score:.1f}/100  "
                  f"(rolling avg: {self.rolling_avg:.1f})")
            print(f"   {result.summary()}")

            # Fire callback for UI streaming
            if self.on_episode:
                self.on_episode({
                    "episode": ep_num,
                    "level": curr["level"],
                    "level_name": curr["name"],
                    "action": task["action"],
                    "score": result.total_score,
                    "rolling_avg": self.rolling_avg,
                    "penalties": result.penalties,
                    "bonuses": result.bonuses,
                    "status": status,
                })

            # Check if target reached
            if (ep_num >= self.config.window_size
                    and self.rolling_avg >= self.config.target_score):
                print(f"\n🎉 Target score {self.config.target_score} reached! "
                      f"Rolling avg: {self.rolling_avg:.1f}")
                break

            # Small delay between episodes
            await asyncio.sleep(0.5)

        self._running = False
        stats = self.memory.stats()
        stats["episodes_run"] = self._episode_count
        stats["final_rolling_avg"] = round(self.rolling_avg, 1)
        stats["level_reached"] = self.current_curriculum_level["name"]
        return stats

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pick_task(self, curriculum_level: dict) -> dict:
        import random
        return random.choice(curriculum_level["tasks"])

    async def _generate(self, task: dict, full_prompt: str) -> dict:
        """
        Call the generation function.
        If no real generator set, uses the built-in pixel art / code generators.
        """
        if self._generate_fn:
            return await self._generate_fn(task, full_prompt)
        return await self._builtin_generate(task, full_prompt)

    async def _builtin_generate(self, task: dict, prompt: str) -> dict:
        """Built-in generation using local pixel artist + code templates."""
        action = task["action"]
        params = task.get("params", {})

        if action in ("draw_character", "draw_tile", "draw_prop"):
            try:
                import sys
                from pathlib import Path as _Path
                sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
                from ai_game_agent.tools.pixel_artist import (
                    draw_character, draw_tile, draw_prop, draw_to_base64
                )
                if action == "draw_character":
                    img = draw_character(
                        archetype=params.get("archetype", "warrior"),
                        direction=params.get("direction", "south"),
                        size=params.get("size", 32),
                    )
                elif action == "draw_tile":
                    img = draw_tile(
                        tile_type=params.get("tile_type", "grass"),
                        size=params.get("size", 16),
                    )
                else:
                    img = draw_prop(
                        prop_type=params.get("prop_type", "tree"),
                        size=params.get("size", 32),
                    )
                b64 = draw_to_base64(img)
                return {"image": b64, "summary": f"{action} {params}"}
            except Exception as e:
                return {"error": str(e), "summary": f"generation failed: {e}"}

        elif "code" in action or "design" in action:
            # Generate code template based on action type
            code = self._code_template(action, params)
            return {"code": code, "description": prompt, "summary": action}

        return {"summary": action, "note": "no generator for this action type"}

    def _code_template(self, action: str, params: dict) -> str:
        """Returns a GDScript template appropriate for the action."""
        complexity = params.get("complexity", "simple")
        if "player" in action:
            return _PLAYER_TEMPLATE_HARD if complexity == "hard" else _PLAYER_TEMPLATE_SIMPLE
        elif "enemy" in action:
            return _ENEMY_TEMPLATE
        elif "scene" in action:
            return _SCENE_TEMPLATE
        elif "design_game" in action:
            return _GAME_DESIGN_TEMPLATE
        else:
            return _PLAYER_TEMPLATE_SIMPLE

    def _evaluate(self, action_type: str, output: dict) -> EvalResult:
        image = output.get("image")
        code  = output.get("code", "")
        desc  = output.get("description", "") + " " + output.get("summary", "")
        return self.evaluator.evaluate(image=image, code=code, description=desc)


# ── GDScript templates ─────────────────────────────────────────────────────────
# Used for training when no LLM is available

_PLAYER_TEMPLATE_SIMPLE = """\
extends CharacterBody2D

class_name Player

var speed := 200.0
var health := 100
var score := 0

func _ready() -> void:
\tadd_to_group("player")

func _physics_process(delta: float) -> void:
\tvar dir := Vector2.ZERO
\tif Input.is_action_pressed("ui_right"): dir.x += 1
\tif Input.is_action_pressed("ui_left"):  dir.x -= 1
\tif Input.is_action_pressed("ui_up"):    dir.y -= 1
\tif Input.is_action_pressed("ui_down"):  dir.y += 1
\tvelocity = dir.normalized() * speed
\tmove_and_slide()

func take_damage(amount: int) -> void:
\thealth -= amount
\tif health <= 0:
\t\tdie()

func die() -> void:
\tget_tree().reload_current_scene()
"""

_PLAYER_TEMPLATE_HARD = """\
extends CharacterBody2D

class_name Player

signal health_changed(new_health: int)
signal died

enum State { IDLE, WALK, RUN, ATTACK, HURT, DEAD }

@export var speed := 200.0
@export var sprint_multiplier := 1.6
@export var max_health := 100
@export var attack_damage := 15

var health := max_health:
\tset(v):
\t\thealth = clamp(v, 0, max_health)
\t\thealth_changed.emit(health)
\t\tif health == 0:
\t\t\t_change_state(State.DEAD)

var score := 0
var level := 1
var xp := 0

var _state := State.IDLE
var _facing := Vector2.DOWN

func _ready() -> void:
\tadd_to_group("player")
\t$AnimationPlayer.play("idle")

func _physics_process(delta: float) -> void:
\tif _state == State.DEAD: return
\tvar dir := Input.get_vector("ui_left","ui_right","ui_up","ui_down")
\tvar sprinting := Input.is_action_pressed("sprint")
\tvar spd := speed * (sprint_multiplier if sprinting else 1.0)
\tif dir != Vector2.ZERO:
\t\t_facing = dir
\t\tvelocity = dir.normalized() * spd
\t\t_change_state(State.RUN if sprinting else State.WALK)
\telse:
\t\tvelocity = velocity.move_toward(Vector2.ZERO, spd * 10 * delta)
\t\t_change_state(State.IDLE)
\tmove_and_slide()

func _input(event: InputEvent) -> void:
\tif event.is_action_pressed("attack"):
\t\t_do_attack()

func _do_attack() -> void:
\tif _state in [State.DEAD, State.ATTACK]: return
\t_change_state(State.ATTACK)
\tvar area := $AttackArea
\tfor body in area.get_overlapping_bodies():
\t\tif body.is_in_group("enemy"):
\t\t\tbody.take_damage(attack_damage)

func take_damage(amount: int) -> void:
\tif _state == State.DEAD: return
\thealth -= amount
\tif health > 0:
\t\t_change_state(State.HURT)

func gain_xp(amount: int) -> void:
\txp += amount
\tif xp >= level * 100:
\t\tlevel += 1
\t\txp = 0
\t\tmax_health += 10
\t\thealth = max_health

func _change_state(new_state: State) -> void:
\tif _state == new_state: return
\t_state = new_state
\tmatch _state:
\t\tState.IDLE:   $AnimationPlayer.play("idle")
\t\tState.WALK:   $AnimationPlayer.play("walk")
\t\tState.RUN:    $AnimationPlayer.play("run")
\t\tState.ATTACK: $AnimationPlayer.play("attack")
\t\tState.HURT:   $AnimationPlayer.play("hurt")
\t\tState.DEAD:
\t\t\t$AnimationPlayer.play("death")
\t\t\tdied.emit()
"""

_ENEMY_TEMPLATE = """\
extends CharacterBody2D

class_name Enemy

signal died(position: Vector2)

enum State { PATROL, CHASE, ATTACK, HURT, DEAD }

@export var speed := 80.0
@export var chase_speed := 130.0
@export var health := 50
@export var attack_damage := 8
@export var detection_range := 200.0
@export var attack_range := 40.0
@export var loot_drop_chance := 0.3

var _state := State.PATROL
var _player: Node = null
var _patrol_dir := Vector2.RIGHT
var _attack_timer := 0.0

func _ready() -> void:
\tadd_to_group("enemy")
\t_player = get_tree().get_first_node_in_group("player")

func _physics_process(delta: float) -> void:
\tif _state == State.DEAD: return
\t_attack_timer = max(0.0, _attack_timer - delta)
\tmatch _state:
\t\tState.PATROL: _patrol(delta)
\t\tState.CHASE:  _chase(delta)
\t\tState.ATTACK: _try_attack()
\tmove_and_slide()

func _patrol(delta: float) -> void:
\tvelocity = _patrol_dir * speed
\tif _player and global_position.distance_to(_player.global_position) < detection_range:
\t\t_change_state(State.CHASE)
\tif is_on_wall():
\t\t_patrol_dir *= -1

func _chase(delta: float) -> void:
\tif not _player: return
\tvar dist := global_position.distance_to(_player.global_position)
\tif dist > detection_range * 1.5:
\t\t_change_state(State.PATROL)
\telif dist < attack_range:
\t\t_change_state(State.ATTACK)
\telse:
\t\tvelocity = (_player.global_position - global_position).normalized() * chase_speed

func _try_attack() -> void:
\tif _attack_timer > 0: return
\t_attack_timer = 1.5
\tif _player:
\t\t_player.take_damage(attack_damage)

func take_damage(amount: int) -> void:
\tif _state == State.DEAD: return
\thealth -= amount
\tif health <= 0:
\t\t_die()
\telse:
\t\t_change_state(State.HURT)

func _die() -> void:
\t_change_state(State.DEAD)
\tdied.emit(global_position)
\tif randf() < loot_drop_chance:
\t\t_drop_loot()
\tqueue_free()

func _drop_loot() -> void:
\tpass  # Spawn item scene at global_position

func _change_state(new_state: State) -> void:
\t_state = new_state
"""

_SCENE_TEMPLATE = """\
extends Node2D

class_name GameScene

@onready var player := $Player
@onready var tilemap := $TileMap
@onready var enemy_spawner := $EnemySpawner
@onready var score_label: Label = $HUD/ScoreLabel
@onready var health_bar := $HUD/HealthBar

var score := 0
var wave := 1

func _ready() -> void:
\tplayer.health_changed.connect(_on_player_health_changed)
\tplayer.died.connect(_on_player_died)
\t_start_wave()

func _start_wave() -> void:
\tenemy_spawner.spawn_wave(wave)
\twait_for_enemies_cleared()

func _on_player_health_changed(new_health: int) -> void:
\thealth_bar.value = new_health

func _on_player_died() -> void:
\t$GameOverScreen.show()
\tget_tree().paused = true

func add_score(amount: int) -> void:
\tscore += amount
\tscore_label.text = "Score: %d" % score

func _on_wave_cleared() -> void:
\twave += 1
\t_start_wave()

func wait_for_enemies_cleared() -> void:
\twhile get_tree().get_nodes_in_group("enemy").size() > 0:
\t\tawait get_tree().process_frame
\t_on_wave_cleared()
"""

_GAME_DESIGN_TEMPLATE = """\
# GAME DESIGN — Isekai Village Sim RPG
# Full game architecture with all major systems

extends Node

class_name GameManager

signal quest_completed(quest_id: String)
signal item_obtained(item_name: String, amount: int)
signal level_up(new_level: int)

# ── Player Data ──────────────────────────────────────────────────────────────
var player_name := "Traveller"
var player_class := "Villager"
var level := 1
var xp := 0
var xp_to_next := 100
var gold := 50

# ── Stats ───────────────────────────────────────────────────────────────────
var stats := {
\t"hp": 100, "max_hp": 100,
\t"mp": 50,  "max_mp": 50,
\t"attack": 10, "defense": 5,
\t"speed": 8, "luck": 3
}

# ── Inventory ────────────────────────────────────────────────────────────────
var inventory: Dictionary = {}  # item_id → count
var equipment: Dictionary = {"weapon": "", "armor": "", "accessory": ""}

# ── Quests ───────────────────────────────────────────────────────────────────
var active_quests: Dictionary = {}
var completed_quests: Array[String] = []

# ── World State ──────────────────────────────────────────────────────────────
var current_map := "village"
var discovered_maps: Array[String] = ["village"]
var npc_states: Dictionary = {}
var day := 1
var time_of_day := 0.0  # 0.0 = dawn, 1.0 = night

func _ready() -> void:
\tload_game()
\t_start_day_cycle()

func _process(delta: float) -> void:
\ttime_of_day += delta / 600.0  # 10-minute day
\tif time_of_day >= 1.0:
\t\ttime_of_day = 0.0
\t\t_advance_day()

func gain_xp(amount: int) -> void:
\txp += amount
\tif xp >= xp_to_next:
\t\t_level_up()

func _level_up() -> void:
\tlevel += 1
\txp -= xp_to_next
\txp_to_next = int(xp_to_next * 1.4)
\tstats["max_hp"] += 10
\tstats["hp"] = stats["max_hp"]
\tstats["attack"] += 2
\tlevel_up.emit(level)

func add_item(item_id: String, amount: int = 1) -> void:
\tinventory[item_id] = inventory.get(item_id, 0) + amount
\titem_obtained.emit(item_id, amount)

func save_game() -> void:
\tvar data := {
\t\t"level": level, "xp": xp, "gold": gold,
\t\t"stats": stats, "inventory": inventory,
\t\t"active_quests": active_quests,
\t\t"completed_quests": completed_quests,
\t\t"current_map": current_map, "day": day
\t}
\tvar file := FileAccess.open("user://save.json", FileAccess.WRITE)
\tfile.store_string(JSON.stringify(data))
\tfile.close()

func load_game() -> void:
\tif not FileAccess.file_exists("user://save.json"): return
\tvar file := FileAccess.open("user://save.json", FileAccess.READ)
\tvar data: Dictionary = JSON.parse_string(file.get_as_text())
\tfile.close()
\tlevel = data.get("level", 1)
\txp = data.get("xp", 0)
\tgold = data.get("gold", 50)
\tstats = data.get("stats", stats)
\tinventory = data.get("inventory", {})
\tcompleted_quests = data.get("completed_quests", [])
\tcurrent_map = data.get("current_map", "village")
\tday = data.get("day", 1)

func _advance_day() -> void:
\tday += 1

func _start_day_cycle() -> void:
\tpass
"""


# ── Convenience runner ────────────────────────────────────────────────────────

async def run_training_session(
    episodes: int = 50,
    target: float = 75.0,
    on_episode: Optional[Callable] = None,
) -> dict:
    cfg = TrainingConfig(max_episodes=episodes, target_score=target)
    trainer = RLTrainer(config=cfg, on_episode=on_episode)
    return await trainer.run()
