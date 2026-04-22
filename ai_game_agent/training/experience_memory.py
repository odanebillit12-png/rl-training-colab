"""
Experience Memory
=================
Stores training episodes (state → action → reward).
Best episodes are replayed as few-shot examples to guide future generation.
Worst episodes are used as negative examples ("don't do this").

Persisted to JSON so training survives restarts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


MEMORY_PATH = Path(__file__).parent.parent / "training_data" / "rl_memory.json"


@dataclass
class Episode:
    episode_id: str
    timestamp: float
    action_type: str          # "draw_character", "generate_code", "design_map", etc.
    action_params: dict       # what was asked
    output_summary: str       # brief description of what was generated
    pixel_art_score: float
    code_score: float
    design_score: float
    total_score: float
    penalties: list[str] = field(default_factory=list)
    bonuses: list[str]   = field(default_factory=list)
    lesson: str = ""          # derived lesson for the agent

    @property
    def is_good(self) -> bool:
        return self.total_score >= 65.0

    @property
    def is_bad(self) -> bool:
        return self.total_score < 40.0


class ExperienceMemory:
    """
    Ring-buffer of training episodes, ranked by score.
    Provides:
      - top_examples(n)   → best n episodes (positive examples)
      - bad_examples(n)   → worst n episodes (negative examples / lessons)
      - few_shot_prompt() → formatted string to inject into LLM context
    """

    MAX_SIZE = 500

    def __init__(self):
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._episodes: list[Episode] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self):
        if MEMORY_PATH.exists():
            try:
                raw = json.loads(MEMORY_PATH.read_text())
                self._episodes = [Episode(**e) for e in raw]
                print(f"[Memory] Loaded {len(self._episodes)} episodes from disk")
            except Exception as e:
                print(f"[Memory] Load failed: {e} — starting fresh")

    def save(self):
        data = [asdict(e) for e in self._episodes]
        MEMORY_PATH.write_text(json.dumps(data, indent=2))

    # ── Add episode ────────────────────────────────────────────────────────────

    def add(
        self,
        action_type: str,
        action_params: dict,
        output_summary: str,
        pixel_art_score: float,
        code_score: float,
        design_score: float,
        total_score: float,
        penalties: list[str],
        bonuses: list[str],
    ) -> Episode:
        lesson = self._derive_lesson(penalties, bonuses, total_score)
        ep = Episode(
            episode_id=f"ep_{int(time.time()*1000)}",
            timestamp=time.time(),
            action_type=action_type,
            action_params=action_params,
            output_summary=output_summary,
            pixel_art_score=pixel_art_score,
            code_score=code_score,
            design_score=design_score,
            total_score=total_score,
            penalties=penalties,
            bonuses=bonuses,
            lesson=lesson,
        )
        self._episodes.append(ep)
        # Keep only MAX_SIZE most recent
        if len(self._episodes) > self.MAX_SIZE:
            self._episodes = sorted(self._episodes, key=lambda e: e.timestamp)[-self.MAX_SIZE:]
        self.save()
        return ep

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def top_examples(self, n: int = 5, action_type: Optional[str] = None) -> list[Episode]:
        eps = self._episodes
        if action_type:
            eps = [e for e in eps if e.action_type == action_type]
        return sorted(eps, key=lambda e: e.total_score, reverse=True)[:n]

    def bad_examples(self, n: int = 5, action_type: Optional[str] = None) -> list[Episode]:
        eps = self._episodes
        if action_type:
            eps = [e for e in eps if e.action_type == action_type]
        return sorted(eps, key=lambda e: e.total_score)[:n]

    def stats(self) -> dict:
        if not self._episodes:
            return {"total": 0, "avg_score": 0, "best": 0, "worst": 0, "good_pct": 0}
        scores = [e.total_score for e in self._episodes]
        good = sum(1 for e in self._episodes if e.is_good)
        return {
            "total": len(self._episodes),
            "avg_score": round(sum(scores) / len(scores), 1),
            "best": round(max(scores), 1),
            "worst": round(min(scores), 1),
            "good_pct": round(good / len(self._episodes) * 100, 1),
        }

    def reward_curve(self) -> list[float]:
        """Returns list of total scores in chronological order (for plotting)."""
        sorted_eps = sorted(self._episodes, key=lambda e: e.timestamp)
        return [e.total_score for e in sorted_eps]

    # ── Few-shot prompt builder ────────────────────────────────────────────────

    def few_shot_prompt(self, action_type: Optional[str] = None) -> str:
        """
        Builds a text block the LLM can read to understand what good/bad looks like.
        Inject this into system or user prompt before generation.
        """
        lines = ["=== TRAINING MEMORY ===",
                 "You have learned from past generation attempts. Apply these lessons:\n"]

        # Positive lessons
        good = self.top_examples(3, action_type)
        if good:
            lines.append("✅ WHAT WORKED WELL:")
            for ep in good:
                lines.append(f"  • [{ep.action_type}] Score {ep.total_score:.0f}/100 — {ep.lesson}")
                if ep.bonuses:
                    lines.append(f"    Bonuses: {', '.join(ep.bonuses[:3])}")

        lines.append("")

        # Negative lessons
        bad = self.bad_examples(3, action_type)
        bad = [b for b in bad if b.is_bad]
        if bad:
            lines.append("❌ WHAT TO AVOID (penalised in past):")
            for ep in bad:
                lines.append(f"  • [{ep.action_type}] Score {ep.total_score:.0f}/100 — {ep.lesson}")
                if ep.penalties:
                    lines.append(f"    Penalties: {', '.join(ep.penalties[:3])}")

        lines.append("\n=== END TRAINING MEMORY ===\n")
        return "\n".join(lines)

    # ── Lesson derivation ─────────────────────────────────────────────────────

    def _derive_lesson(self, penalties: list[str], bonuses: list[str], score: float) -> str:
        if score >= 80:
            top = bonuses[0] if bonuses else "high quality output"
            return f"Excellent output ({score:.0f}/100): {top}"
        elif score >= 60:
            return f"Good output ({score:.0f}/100) — keep similar approach"
        elif score >= 40:
            tip = penalties[0] if penalties else "needs improvement"
            return f"Average output ({score:.0f}/100) — fix: {tip}"
        else:
            tip = penalties[0] if penalties else "major quality issues"
            return f"PENALISED ({score:.0f}/100) — avoid: {tip}"
