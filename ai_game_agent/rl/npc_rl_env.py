"""
NPC RL Environment — enemies, bosses, and allies that learn from combat.

Each NPC type has its own PPO policy trained against simulated players.
Bosses get extra training and remember how they died last time.

State (24 floats):
  - 3: NPC position normalized (x, y, facing)
  - 3: Player position (x, y, distance)
  - 4: Health (npc_hp, player_hp, npc_hp_pct, player_hp_pct)
  - 3: Combat context (time_since_last_hit, player_is_attacking, player_in_range)
  - 4: NPC status (stamina, cooldown_pct, is_stunned, aggro_level)
  - 4: Recent history (last_action, last_player_action, hits_taken, hits_dealt)
  - 3: Environment (phase_of_boss_fight, player_direction, time_in_fight)

Actions (8):
  0 = WAIT           stay in place, observe
  1 = MOVE_TOWARD    advance on player
  2 = MOVE_AWAY      retreat / reposition
  3 = ATTACK_MELEE   basic attack
  4 = ATTACK_RANGED  projectile / spell
  5 = SPECIAL        signature move (each monster type defines this)
  6 = DEFEND         block / parry
  7 = HEAL           use potion or regen (boss only)

Reward shaping:
  +2.0  deal damage to player
  +0.5  player near death
  +1.0  successful block/dodge
  -1.5  take damage
  -3.0  NPC dies
  +5.0  player dies (episode win)
  +0.1  time spent in combat range (aggression bonus)
  -0.05 per step standing still (discourages passive AI)

Exported as ONNX → loaded in Godot via:
  var model = ONNXModel.new()
  model.load("res://ai/enemies/goblin.onnx")
  var probs = model.run([state_vector])
  var action = probs.argmax()
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict

NPC_ACTION_NAMES = [
    "WAIT", "MOVE_TOWARD", "MOVE_AWAY",
    "ATTACK_MELEE", "ATTACK_RANGED", "SPECIAL",
    "DEFEND", "HEAL",
]

OBS_DIM    = 24
ACTION_DIM = 8

# ── NPC type presets ──────────────────────────────────────────────────────────
# Each preset shapes rewards differently so each monster type has unique behaviour

NPC_PRESETS: Dict[str, dict] = {
    "goblin": {
        "hp": 30, "atk": 8, "spd": 1.4,
        "special": "stab_flurry",
        "aggression": 1.5,   # rushes player
        "defend_bonus": 0.5,
    },
    "knight": {
        "hp": 80, "atk": 15, "spd": 0.8,
        "special": "shield_bash",
        "aggression": 0.8,
        "defend_bonus": 2.0,  # rewards blocking
    },
    "mage": {
        "hp": 40, "atk": 25, "spd": 1.0,
        "special": "fireball_barrage",
        "aggression": 0.6,   # stays at range
        "ranged_bonus": 2.0,
    },
    "dragon_boss": {
        "hp": 500, "atk": 40, "spd": 0.9,
        "special": "breath_weapon",
        "aggression": 1.2,
        "phase_transitions": [0.7, 0.4],  # enrage at 70% and 40% hp
        "memory": True,      # remembers how player killed it before
    },
    "lich_boss": {
        "hp": 350, "atk": 35, "spd": 1.0,
        "special": "soul_drain",
        "aggression": 0.9,
        "ranged_bonus": 1.5,
        "phase_transitions": [0.5],
        "memory": True,
    },
    "slime": {
        "hp": 20, "atk": 5, "spd": 0.6,
        "special": "split",
        "aggression": 0.7,
        "defend_bonus": 0.2,
    },
}


class NPCEnv:
    """
    Gym-style environment simulating NPC vs Player combat.

    One episode = one fight from start to either NPC death or player death.
    The player is simulated with a fixed heuristic policy (attack, dodge, heal).
    """

    def __init__(self, npc_type: str = "goblin", max_steps: int = 200):
        self.npc_type  = npc_type
        self.preset    = NPC_PRESETS.get(npc_type, NPC_PRESETS["goblin"])
        self.max_steps = max_steps
        self.obs_dim   = OBS_DIM
        self.action_dim = ACTION_DIM

        # Combat memory for bosses (tracks player strategy)
        self._player_action_history: list = []
        self._deaths_by_player: int = 0

        self._reset_state()

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = 0.0

        # ── NPC acts ──────────────────────────────────────────────────────────
        dmg_to_player = 0.0
        dmg_to_npc    = 0.0

        if action == 0:   # WAIT
            reward -= 0.05
        elif action == 1: # MOVE_TOWARD
            self._npc_distance = max(1.0, self._npc_distance - self.preset["spd"])
            reward += 0.05 * self.preset["aggression"]
        elif action == 2: # MOVE_AWAY
            self._npc_distance = min(20.0, self._npc_distance + self.preset["spd"])
        elif action == 3: # ATTACK_MELEE
            if self._npc_distance <= 3.0 and self._atk_cooldown <= 0:
                dmg_to_player = self.preset["atk"] * np.random.uniform(0.8, 1.2)
                self._atk_cooldown = 3
                reward += 2.0
        elif action == 4: # ATTACK_RANGED
            if self._atk_cooldown <= 0:
                bonus  = self.preset.get("ranged_bonus", 1.0)
                dmg_to_player = self.preset["atk"] * 0.7 * np.random.uniform(0.9, 1.1)
                self._atk_cooldown = 4
                reward += 1.5 * bonus
        elif action == 5: # SPECIAL
            if self._special_cooldown <= 0:
                dmg_to_player = self.preset["atk"] * 1.8 * np.random.uniform(0.9, 1.1)
                self._special_cooldown = 15
                reward += 3.0
        elif action == 6: # DEFEND
            self._is_defending = True
            reward += 0.3 * self.preset.get("defend_bonus", 1.0)
        elif action == 7: # HEAL (boss only)
            if self._npc_hp < self.preset["hp"] * 0.3:
                heal = self.preset["hp"] * 0.1
                self._npc_hp = min(self.preset["hp"], self._npc_hp + heal)
                reward += 0.5

        # ── Player acts (heuristic opponent) ─────────────────────────────────
        player_action = self._simulate_player()
        self._player_action_history.append(player_action)
        if player_action == "attack" and not self._is_defending:
            base_dmg   = 15.0 * np.random.uniform(0.8, 1.2)
            dmg_to_npc = base_dmg
        elif player_action == "attack" and self._is_defending:
            dmg_to_npc = 5.0  # reduced damage when defending
            reward += 1.0     # successful block

        # ── Apply damage ──────────────────────────────────────────────────────
        if dmg_to_player > 0:
            self._player_hp -= dmg_to_player
            self._hits_dealt += 1
            if self._player_hp < self.preset["hp"] * 0.3:
                reward += 0.5  # bonus: player near death

        if dmg_to_npc > 0:
            self._npc_hp -= dmg_to_npc
            self._hits_taken += 1
            reward -= 1.5

        # ── Cooldowns ─────────────────────────────────────────────────────────
        self._atk_cooldown     = max(0, self._atk_cooldown - 1)
        self._special_cooldown = max(0, self._special_cooldown - 1)
        self._is_defending     = False
        self._time_in_fight   += 1

        # ── Check boss phase transitions ──────────────────────────────────────
        hp_pct = self._npc_hp / self.preset["hp"]
        for phase_thresh in self.preset.get("phase_transitions", []):
            if hp_pct < phase_thresh and self._phase < self.preset.get("phase_transitions", []).index(phase_thresh) + 1:
                self._phase += 1
                reward += 1.0  # entered new phase

        # ── Terminal conditions ───────────────────────────────────────────────
        self._step += 1
        done   = False
        result = "ongoing"

        if self._player_hp <= 0:
            reward += 5.0
            done   = True
            result = "npc_wins"
        elif self._npc_hp <= 0:
            reward -= 3.0
            self._deaths_by_player += 1
            done   = True
            result = "player_wins"
        elif self._step >= self.max_steps:
            done   = True
            result = "timeout"

        obs = self._get_obs()
        return obs, reward, done, {
            "result": result,
            "npc_hp": self._npc_hp,
            "player_hp": self._player_hp,
            "hits_dealt": self._hits_dealt,
            "hits_taken": self._hits_taken,
        }

    def get_player_counter_tendencies(self) -> Dict[str, float]:
        """
        Analyse what the player tends to do — bosses use this to adapt.
        Returns action frequencies for adaptive difficulty.
        """
        if not self._player_action_history:
            return {}
        total = len(self._player_action_history)
        actions = ["attack", "dodge", "heal", "retreat"]
        return {a: self._player_action_history.count(a) / total for a in actions}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_state(self):
        self._npc_hp          = float(self.preset["hp"])
        self._player_hp       = 100.0
        self._npc_distance    = 10.0
        self._atk_cooldown    = 0
        self._special_cooldown= 0
        self._is_defending    = False
        self._hits_dealt      = 0
        self._hits_taken      = 0
        self._time_in_fight   = 0
        self._phase           = 0
        self._step            = 0
        self._last_action     = 0
        self._last_player_action = 0

    def _get_obs(self) -> np.ndarray:
        hp_max    = float(self.preset["hp"])
        npc_hp_pct = self._npc_hp / hp_max
        plr_hp_pct = self._player_hp / 100.0
        dist_norm  = self._npc_distance / 20.0
        phase_norm = self._phase / max(1, len(self.preset.get("phase_transitions", [1])))

        obs = np.array([
            # Position
            0.5, 0.5, 0.0,                          # NPC pos (normalized)
            0.5 - dist_norm, 0.5, dist_norm,         # Player pos + dist

            # Health
            self._npc_hp / hp_max,
            self._player_hp / 100.0,
            npc_hp_pct,
            plr_hp_pct,

            # Combat context
            min(1.0, self._atk_cooldown / 5.0),
            float(self._npc_distance <= 4.0),       # player in melee range
            float(self._npc_distance <= 10.0),      # player in ranged range

            # NPC status
            1.0,                                     # stamina (full)
            min(1.0, self._special_cooldown / 15.0),
            0.0,                                     # stunned
            min(1.0, float(self._hits_taken) / 10), # aggro level

            # History
            float(self._last_action) / ACTION_DIM,
            float(self._last_player_action) / 4.0,
            min(1.0, float(self._hits_taken) / 20),
            min(1.0, float(self._hits_dealt) / 20),

            # Environment
            phase_norm,
            float(self._npc_distance > self._npc_distance),  # player moving toward
            min(1.0, float(self._time_in_fight) / self.max_steps),
            float(self.preset.get("memory", False)),
        ], dtype=np.float32)
        return obs

    def _simulate_player(self) -> str:
        """Fixed heuristic player: attacks when in range, heals when low, else advances."""
        # If boss has memory, player uses a counter-strategy
        if self.preset.get("memory") and self._deaths_by_player > 0:
            tendencies = self.get_player_counter_tendencies()
            # Player adapts based on what worked before
            if tendencies.get("attack", 0) > 0.6:
                return "dodge" if np.random.random() < 0.3 else "attack"

        if self._player_hp < 30 and np.random.random() < 0.4:
            return "heal"
        if self._npc_distance <= 3.0:
            return "attack"
        return "advance"
