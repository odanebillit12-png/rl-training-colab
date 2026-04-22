"""
World Builder Trainer
=====================
Trains AI Dev to design GOTY-quality game worlds.

Each episode, AI Dev:
  1. Picks a world-building CHALLENGE (from a curriculum)
  2. Generates a world design response (description + systems + code snippets)
  3. Gets scored by GOTYEvaluator on all 10 dimensions
  4. Learns from the feedback — best designs become few-shot examples

Curriculum levels:
  Level 1 — Concept: describe a single biome with 3 NPC archetypes
  Level 2 — Region: design a full region (biome + dungeon + village + lore)
  Level 3 — World:  design a complete world (5+ regions + overarching narrative)
  Level 4 — GOTY:   full game design doc with all systems, feel notes, art direction
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .goty_evaluator import GOTYEvaluator, GOTYResult, DIMENSION_LABELS, GRADE_LABELS
from .experience_memory import ExperienceMemory


# ── Curriculum challenges ────────────────────────────────────────────────────

WORLD_CHALLENGES = {
    1: [  # Concept level — single element design
        {
            "name": "Village Design",
            "prompt": (
                "Design a starting village for an Isekai RPG. Include: village name, "
                "3-5 NPC inhabitants with names/jobs/personalities/schedules, "
                "2-3 quests available, resources available, art style notes, "
                "and what makes this village memorable/emotional."
            ),
            "hint": "Think Stardew Valley — every NPC should feel like a real person.",
        },
        {
            "name": "Dungeon Design",
            "prompt": (
                "Design one dungeon for an Isekai RPG. Include: dungeon name/lore, "
                "3 room types with distinct purposes, boss with backstory and attack patterns, "
                "unique mechanic that makes this dungeon feel different, "
                "rewards and why they matter to the player."
            ),
            "hint": "Think Hollow Knight — the dungeon should tell its own story through its layout.",
        },
        {
            "name": "Magic System",
            "prompt": (
                "Design a magic system for an Isekai RPG. Include: source of magic power, "
                "6-8 spell types with clear mechanics, costs/limitations that create strategy, "
                "how magic interacts with the world/NPCs/economy, "
                "visual and audio feedback for each spell type."
            ),
            "hint": "Think CrossCode — systems should be deep but learnable, with combo potential.",
        },
    ],
    2: [  # Region level
        {
            "name": "Region Design",
            "prompt": (
                "Design a complete game region for an Isekai RPG. Include:\n"
                "- Region name, biome, atmosphere, color palette\n"
                "- 3 sub-areas (e.g. village, dungeon, wilderness) with distinct purposes\n"
                "- 5 named NPCs with schedules, motivations, and relationships to each other\n"
                "- Main quest arc + 2 side quests rooted in regional lore\n"
                "- Unique resource/mechanic only in this region\n"
                "- How the region connects to the wider world narrative\n"
                "- Music theme description and visual storytelling details"
            ),
            "hint": "Each region should feel like it could be its own game. Stardew/Hades level NPC depth.",
        },
        {
            "name": "Game Feel Design",
            "prompt": (
                "Design the complete game feel and juice system for an Isekai RPG. Include:\n"
                "- Player movement: walk speed, run speed, jump feel, coyote time, input buffer\n"
                "- Combat feedback: hit stop frames, screen shake values, particle effects\n"
                "- Audio feedback: every action type and its sound cue\n"
                "- Camera behavior: follow speed, zoom levels, shake triggers\n"
                "- Transitions: how scenes change, how menus open, how dialogue flows\n"
                "- Death/respawn feel: what happens, how it feels emotionally\n"
                "Provide specific GDScript code for the hit_stop and screen_shake systems."
            ),
            "hint": "Hades is the gold standard. Every hit should FEEL impactful.",
        },
    ],
    3: [  # World level
        {
            "name": "Full World Design",
            "prompt": (
                "Design the complete world of an Isekai RPG called 'Chronicles of New World'. Include:\n"
                "- World overview: size, tone, themes, what makes it unique vs other isekai\n"
                "- 5 distinct regions: name, biome, primary conflict, key NPC, unique mechanic\n"
                "- Overarching narrative: inciting incident, rising action, climax, resolution\n"
                "- 3 major factions with relationships, goals, and how player can influence them\n"
                "- Progression arc: how player grows from start to end, what changes in the world\n"
                "- Replayability hooks: what changes on a second playthrough\n"
                "- Art direction: consistent pixel art style that ties all regions together\n"
                "- The ONE thing players will remember and tell friends about"
            ),
            "hint": "What's the Undertale moment? The Hades revelation? The Stardew connection? Design for memory.",
        },
    ],
    4: [  # GOTY level — complete GDD
        {
            "name": "Complete Game Design Document",
            "prompt": (
                "Write a complete Game Design Document for 'Isekai Chronicles of New World' — "
                "targeting Game of the Year quality. The GDD must cover:\n\n"
                "VISION: Elevator pitch, target audience, comparable titles, unique selling point\n"
                "WORLD: 5 regions, full narrative arc, faction system, lore timeline\n"
                "PLAYER: Character classes, progression, agency, customization\n"
                "SYSTEMS: Combat, magic, crafting, economy, NPC relationships, base-building\n"
                "GAME FEEL: Input response, feedback systems, camera, juice, audio design\n"
                "ART DIRECTION: Pixel art style guide, palette, animation principles\n"
                "TECHNICAL: Performance targets, save system, accessibility, platform targets\n"
                "REPLAYABILITY: NG+, secrets, branching, challenge modes\n"
                "EMOTIONAL ARC: The 5 moments players will never forget\n\n"
                "Be specific. No vague statements. Every system should have a clear mechanic."
            ),
            "hint": "This is your GOTY pitch. Make every sentence earn its place.",
        },
    ],
}


# ── LLM World Design Generator ───────────────────────────────────────────────

async def _call_llm(prompt: str, api_keys: dict) -> str:
    """Call available LLM to generate world design response."""
    # Try Groq first (fastest, free)
    if api_keys.get("groq"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_keys['groq']}"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a world-class game designer who has shipped "
                                    "GOTY-winning RPGs. You design games that are emotionally "
                                    "resonant, mechanically deep, and visually cohesive. "
                                    "You know Stardew Valley, Hades, Hollow Knight, CrossCode, "
                                    "and Undertale inside out. Every design decision you make "
                                    "is intentional and serves the player experience. "
                                    "Be specific — no vague statements."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 2000,
                        "temperature": 0.85,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    Groq error: {e}")

    # Try Gemini
    if api_keys.get("gemini"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-2.0-flash-exp:generateContent?key={api_keys['gemini']}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.85},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"    Gemini error: {e}")

    # Fallback — structured template response
    return _template_response(prompt)


def _template_response(prompt: str) -> str:
    """Minimal template when all APIs are rate-limited."""
    return (
        "WORLD DESIGN — Chronicles of New World\n\n"
        "Village: Heartwood — starting settlement with farming, crafting, and 5 NPCs\n"
        "NPC Schedules: villagers wake at 6am, work until 6pm, socialize until 10pm\n"
        "Magic System: element-based with status effects (burn, freeze, poison, stun)\n"
        "Combat Feel: hitstop 3 frames, screen shake 0.1s, particle burst on kill\n"
        "Dungeon: Ancient Ruins with 3 floors, puzzle mechanic, boss with 3 phases\n"
        "Progression: level up system, skill tree, crafting recipes unlock on discovery\n"
        "Faction: Kingdom, Rebels, Ancient Order — player reputation affects quests\n"
        "Save system: autosave on zone change, manual save at inns\n"
        "Replayability: NG+ with harder enemies, hidden true ending, speedrun mode\n"
        "Emotional hook: NPC you helped in act 1 sacrifices themselves in act 3\n"
    )


# ── World Trainer ─────────────────────────────────────────────────────────────

@dataclass
class WorldTrainerConfig:
    max_episodes: int = 30
    target_score: float = 80.0
    api_keys: dict = field(default_factory=dict)
    save_dir: str = "training_data/world_designs"
    curriculum_level: int = 1      # auto-advances based on score


@dataclass
class WorldEpisodeResult:
    episode: int = 0
    challenge_name: str = ""
    level: int = 1
    response: str = ""
    goty_result: Optional[GOTYResult] = None
    score: float = 0.0
    duration_s: float = 0.0


class WorldBuilderTrainer:
    """
    Trains AI Dev on GOTY-quality world design.
    Produces world descriptions that GOTYEvaluator scores.
    Memory stores best designs for few-shot learning.
    """

    def __init__(self, config: WorldTrainerConfig):
        self.cfg = config
        self.evaluator = GOTYEvaluator()
        self.memory = ExperienceMemory()
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._scores: list[float] = []

    async def run(self) -> dict:
        print(f"\n🌍 WORLD BUILDER TRAINING — Target: {self.cfg.target_score}/100")
        print(f"   Curriculum level: {self.cfg.curriculum_level}")
        print(f"   Max episodes: {self.cfg.max_episodes}\n")

        best_score = 0.0
        for ep in range(1, self.cfg.max_episodes + 1):
            result = await self._run_episode(ep)
            self._scores.append(result.score)
            best_score = max(best_score, result.score)

            # Advance curriculum level on consistent good performance
            if len(self._scores) >= 5:
                recent_avg = sum(self._scores[-5:]) / 5
                if recent_avg >= self.cfg.target_score and self.cfg.curriculum_level < 4:
                    self.cfg.curriculum_level += 1
                    print(f"\n  🎓 CURRICULUM ADVANCE → Level {self.cfg.curriculum_level}!\n")

        avg = sum(self._scores) / len(self._scores) if self._scores else 0
        return {
            "best_score": best_score,
            "avg_score": avg,
            "episodes": len(self._scores),
            "final_level": self.cfg.curriculum_level,
        }

    async def _run_episode(self, ep: int) -> WorldEpisodeResult:
        level = self.cfg.curriculum_level
        challenges = WORLD_CHALLENGES.get(level, WORLD_CHALLENGES[1])
        challenge = random.choice(challenges)

        print(f"  Episode {ep:3d} | Level {level} | {challenge['name']}")
        t0 = time.time()

        # Get few-shot examples from memory for this episode
        few_shot = self.memory.get_few_shot_prompt("world_design", top_k=2)

        # Build full prompt with context
        full_prompt = self._build_prompt(challenge, few_shot)

        # Generate response via LLM
        response = await _call_llm(full_prompt, self.cfg.api_keys)

        # Evaluate with GOTY rubric
        goty_result = self.evaluator.evaluate(
            world_description=response,
            design_doc=challenge["prompt"],
        )
        score = goty_result.total
        duration = time.time() - t0

        # Store in memory
        self.memory.add(
            action_type="world_design",
            action_desc=challenge["prompt"][:200],
            result=response[:500],
            score=score,
            metadata={"level": level, "challenge": challenge["name"]},
        )

        # Save best designs to disk
        if score >= 70:
            self._save_design(ep, challenge["name"], response, goty_result)

        # Print result
        grade_label = next(
            (label for threshold, label in GRADE_LABELS if score >= threshold),
            "🌱 EARLY CONCEPT"
        )
        print(f"           Score: {score:.1f}/100  {grade_label}")

        # Show top gap
        if goty_result.next_targets:
            print(f"           Fix:   {goty_result.next_targets[0][:80]}")

        return WorldEpisodeResult(
            episode=ep,
            challenge_name=challenge["name"],
            level=level,
            response=response,
            goty_result=goty_result,
            score=score,
            duration_s=duration,
        )

    def _build_prompt(self, challenge: dict, few_shot: str) -> str:
        parts = []
        if few_shot:
            parts.append(f"REFERENCE EXAMPLES (learn from these):\n{few_shot}\n---\n")
        parts.append(
            f"CHALLENGE: {challenge['name']}\n\n"
            f"{challenge['prompt']}\n\n"
            f"DESIGNER NOTE: {challenge['hint']}\n\n"
            f"Be specific, detailed, and design for GOTY quality."
        )
        return "\n".join(parts)

    def _save_design(self, ep: int, name: str, response: str, result: GOTYResult):
        safe_name = name.lower().replace(" ", "_")
        filepath = self.save_dir / f"ep{ep:03d}_{safe_name}_{result.total:.0f}.txt"
        filepath.write_text(
            f"EPISODE {ep} — {name}\nSCORE: {result.total:.1f}/100\n"
            f"{result.grade}\n\n"
            f"{'='*60}\n{response}\n{'='*60}\n\n"
            f"GOTY BREAKDOWN:\n" +
            "\n".join(f"  {k}: {v*10:.0f}/100" for k, v in result.scores.items())
        )
