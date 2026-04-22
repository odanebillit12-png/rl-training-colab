"""
ONNX Exporter + Godot integration guide for trained NPC policies.

After training, call export_all_npc_policies() to write ONNX files
into godot_ai_colony/ai/enemies/ for direct Godot loading.

Godot usage (GDScript):
    extends CharacterBody2D

    var model: Resource  # ONNXModel resource
    var obs: PackedFloat32Array

    func _ready():
        model = load("res://ai/enemies/goblin.onnx")

    func decide_action(state: Array) -> int:
        obs = PackedFloat32Array(state)
        var probs: PackedFloat32Array = model.run([obs])[0]
        return probs.find(probs.max())  # argmax
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional


def export_all_npc_policies(
    trained_agents: Dict[str, object],   # {npc_type: PPOAgent}
    output_dir: str = "godot_ai_colony/ai/enemies",
) -> Dict[str, str]:
    """
    Export all trained NPC policies to ONNX files.
    Returns {npc_type: onnx_path}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, str] = {}

    for npc_type, agent in trained_agents.items():
        path = str(out / f"{npc_type}.onnx")
        try:
            agent.export_onnx(path, agent_name=npc_type)
            exported[npc_type] = path

            # Write companion GDScript controller
            _write_gdscript_controller(npc_type, path, out)
        except Exception as e:
            print(f"  ⚠️  ONNX export failed for {npc_type}: {e}")

    return exported


def _write_gdscript_controller(npc_type: str, onnx_path: str, out_dir: Path):
    """Write a ready-to-use GDScript AI controller for this NPC type."""
    script = f'''extends CharacterBody2D
## {npc_type.replace("_", " ").title()} AI Controller
## Auto-generated — policy trained via PPO reinforcement learning
## Actions: WAIT=0 MOVE_TOWARD=1 MOVE_AWAY=2 ATTACK_MELEE=3
##          ATTACK_RANGED=4 SPECIAL=5 DEFEND=6 HEAL=7

const ACTION_NAMES = ["WAIT","MOVE_TOWARD","MOVE_AWAY","ATTACK_MELEE",
                      "ATTACK_RANGED","SPECIAL","DEFEND","HEAL"]

@export var move_speed: float = 150.0
@export var attack_range: float = 80.0
@export var ranged_range: float = 300.0

var _model          # ONNXModel (load via plugin)
var _player: Node2D = null
var _hp:         float = 100.0
var _max_hp:     float = 100.0
var _atk_cooldown:   float = 0.0
var _spec_cooldown:  float = 0.0
var _hits_dealt:     int   = 0
var _hits_taken:     int   = 0
var _time_in_fight:  float = 0.0
var _phase:          int   = 0
var _last_action:    int   = 0


func _ready():
    _player = get_tree().get_first_node_in_group("player")
    # Uncomment when Godot ONNX plugin is installed:
    # _model = load("res://ai/enemies/{npc_type}.onnx")


func _physics_process(delta: float):
    if not _player:
        return
    _time_in_fight += delta
    _atk_cooldown  = max(0.0, _atk_cooldown - delta)
    _spec_cooldown = max(0.0, _spec_cooldown - delta)

    var action = _decide_action()
    _execute_action(action, delta)
    _last_action = action


func _decide_action() -> int:
    if not _model:
        return _heuristic_fallback()
    var state = _build_state()
    var probs: PackedFloat32Array = _model.run([PackedFloat32Array(state)])[0]
    var best_idx = 0
    var best_val = probs[0]
    for i in range(1, probs.size()):
        if probs[i] > best_val:
            best_val = probs[i]
            best_idx = i
    return best_idx


func _build_state() -> Array:
    var dist = global_position.distance_to(_player.global_position)
    var hp_pct  = _hp / _max_hp
    var p_hp    = _player.get("hp") if _player.get("hp") else 100.0
    var p_hp_pct = p_hp / 100.0
    return [
        global_position.x / 1920.0, global_position.y / 1080.0, 0.0,
        _player.global_position.x / 1920.0, _player.global_position.y / 1080.0,
        dist / 500.0,
        hp_pct, p_hp_pct, hp_pct, p_hp_pct,
        _atk_cooldown / 0.5, float(dist <= attack_range), float(dist <= ranged_range),
        1.0, _spec_cooldown / 5.0, 0.0, float(_hits_taken) / 10.0,
        float(_last_action) / 8.0, 0.0,
        min(1.0, float(_hits_taken) / 20.0), min(1.0, float(_hits_dealt) / 20.0),
        float(_phase) / 3.0, 0.0,
        min(1.0, _time_in_fight / 60.0),
    ]


func _execute_action(action: int, delta: float):
    match action:
        0: pass  # WAIT
        1: _move_toward_player(delta)
        2: _move_away_player(delta)
        3: _attack_melee()
        4: _attack_ranged()
        5: _special_attack()
        6: _defend()
        7: _heal()


func _move_toward_player(delta):
    var dir = (_player.global_position - global_position).normalized()
    velocity = dir * move_speed
    move_and_slide()


func _move_away_player(delta):
    var dir = (global_position - _player.global_position).normalized()
    velocity = dir * move_speed * 0.7
    move_and_slide()


func _attack_melee():
    if _atk_cooldown <= 0 and global_position.distance_to(_player.global_position) <= attack_range:
        if _player.has_method("take_damage"):
            _player.take_damage(15)
        _hits_dealt += 1
        _atk_cooldown = 0.5


func _attack_ranged():
    if _atk_cooldown <= 0:
        # Spawn projectile (implement per monster type)
        _atk_cooldown = 0.8


func _special_attack():
    if _spec_cooldown <= 0:
        # Signature move (implement per monster type)
        _spec_cooldown = 5.0


func _defend():
    pass  # Set defense flag — reduce incoming damage


func _heal():
    _hp = min(_max_hp, _hp + _max_hp * 0.1)


func take_damage(amount: float):
    _hp -= amount
    _hits_taken += 1
    if _hp <= 0:
        queue_free()


func _heuristic_fallback() -> int:
    var dist = global_position.distance_to(_player.global_position)
    if dist <= attack_range and _atk_cooldown <= 0:
        return 3  # ATTACK_MELEE
    elif dist > attack_range:
        return 1  # MOVE_TOWARD
    return 0  # WAIT
'''
    script_path = out_dir / f"{npc_type}_ai_controller.gd"
    script_path.write_text(script)
    print(f"  📝 GDScript → {script_path}")


def generate_godot_ai_readme(output_dir: str = "godot_ai_colony/ai/enemies") -> str:
    """Generate README explaining how to use the trained policies in Godot."""
    readme = """# NPC AI — Reinforcement Learning Policies

These ONNX models are trained via PPO reinforcement learning.
Each enemy type learned its own fighting style through thousands of simulated battles.

## How to use in Godot

1. Install the **GodotONNX** plugin (AssetLib or manual):
   https://github.com/godot-onnxruntime/godot-onnxruntime

2. Attach the companion GDScript to your enemy scene:
   e.g. `goblin_ai_controller.gd` → attach to `Goblin` CharacterBody2D

3. The AI runs automatically in `_physics_process` — no extra code needed.

## Enemy policies included

| File | Enemy | Special Move |
|------|-------|-------------|
| goblin.onnx | Goblin | Stab Flurry |
| knight.onnx | Knight | Shield Bash |
| mage.onnx | Dark Mage | Fireball Barrage |
| dragon_boss.onnx | Dragon Boss | Breath Weapon |
| lich_boss.onnx | Lich | Soul Drain |

## Boss adaptive memory

Dragon Boss and Lich Boss **remember** how many times the player has killed them.
Each death makes the boss smarter. After 3 defeats, the boss enters "enrage" mode.

## Retraining

Run `python3 -m ai_game_agent.training.rl_trainer` to retrain all policies.
New ONNX files will be exported automatically to this directory.
"""
    path = Path(output_dir) / "README.md"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(readme)
    return str(path)
