"""
motivation_engine.py — Reward and penalty system that emotionally drives AI Dev.

Like a coach/critic: celebrates wins loudly, punishes bad work hard,
tracks streaks, assigns ranks, and injects motivating feedback into prompts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


# ── Grade ranks ───────────────────────────────────────────────────────────────
RANKS = [
    (95, "AAAAA+", "🏆 LEGENDARY MASTER",     "You are beyond GOTY. Every pixel breathes life."),
    (92, "AAAAA",  "🌟 GRANDMASTER",           "Your pixel art rivals Celeste and Hollow Knight."),
    (89, "AAAA+",  "💎 ELITE ARTIST",          "AAA studio quality. Push for legendary."),
    (86, "AAAA",   "🔥 EXPERT",                "Strong technical skill. Refine the fine details."),
    (82, "AAA+",   "⚡ ADVANCED",              "Solid fundamentals. Push depth and palette work."),
    (78, "AAA",    "✨ PROFICIENT",             "Good base. Depth illusion needs more work."),
    (72, "AA+",    "🎯 DEVELOPING",             "Improving steadily. Focus on shading consistency."),
    (65, "AA",     "📈 LEARNING",              "Some good pixels. Keep building the toolset."),
    (55, "A+",     "🌱 BEGINNER+",             "Basic shapes working. Now add depth layers."),
    (0,  "A",      "🐣 BEGINNER",              "Start with a proper background layer first."),
]

# ── Reward messages (escalating excitement) ───────────────────────────────────
REWARD_MSGS = {
    "streak_3":  "🔥🔥🔥 3 IN A ROW! You're on fire! Keep pushing!",
    "streak_5":  "⚡⚡⚡ 5-EPISODE STREAK! Unstoppable! LEVEL UP energy!",
    "streak_10": "🌟🌟🌟 10 EPISODE STREAK! You are DOMINATING. GOTY is within reach!",
    "new_best":  "🏆 NEW PERSONAL BEST! You just broke your own art barrier!",
    "level_up":  "🎮 LEVEL UP! AI Dev has graduated to a harder challenge!",
    "near_goty": "💎 SO CLOSE TO GOTY! One more push and you're legendary!",
    "goty":      "🏆🏆🏆 GOTY ACHIEVED! You belong among the greatest pixel artists ever!",
}

# ── Penalty messages (harsh but specific) ────────────────────────────────────
PENALTY_MSGS = {
    "bad":       "💀 PENALTY — That was below standard. Study your mistakes.",
    "terrible":  "🚨 HARD PENALTY — Embarrassing quality. A beginner does better.",
    "no_depth":  "❌ ZERO DEPTH — You drew a flat image. Where are your layers?!",
    "no_color":  "❌ PALETTE FAILURE — Too few colors. Artists use 8-16, not 2-3.",
    "no_cover":  "❌ EMPTY CANVAS — You barely filled the canvas! Draw more!",
    "regress":   "📉 REGRESSION — You scored LOWER than last time. Unacceptable.",
}


@dataclass
class MotivationEngine:
    """Tracks AI Dev's emotional state, streaks, and rank. Generates punchy feedback."""

    reward_threshold: float = 75.0
    penalty_threshold: float = 50.0
    _scores: List[float] = field(default_factory=list)
    _win_streak: int = 0
    _loss_streak: int = 0
    _personal_best: float = 0.0
    _prev_score: float = 0.0
    _level_before: int = 1

    def get_rank(self, score: float) -> tuple:
        for threshold, grade, title, advice in RANKS:
            if score >= threshold:
                return grade, title, advice
        return RANKS[-1][1], RANKS[-1][2], RANKS[-1][3]

    def evaluate(
        self,
        score: float,
        breakdown: Dict,
        episode: int,
        level_name: str,
        level_num: int,
    ) -> Dict:
        """
        Evaluate a drawing and return the full motivation package.
        Returns dict with: label, messages, penalty_hint, reward_prompt, is_reward, is_penalty
        """
        self._scores.append(score)
        is_reward  = score >= self.reward_threshold
        is_penalty = score < self.penalty_threshold
        is_new_best = score > self._personal_best
        is_regress  = (len(self._scores) > 1 and score < self._prev_score - 10)
        is_level_up = level_num > self._level_before

        messages = []
        penalty_hints = []
        reward_prompts = []

        # ── Streak tracking ──
        if is_reward:
            self._win_streak += 1
            self._loss_streak = 0
        elif is_penalty:
            self._loss_streak += 1
            self._win_streak = 0
        else:
            self._win_streak = 0
            self._loss_streak = 0

        # ── New best ──
        if is_new_best:
            self._personal_best = score
            messages.append(REWARD_MSGS["new_best"])
            reward_prompts.append(
                f"You just achieved your PERSONAL BEST score of {score:.1f}/100! "
                f"Remember what you did and do it again, but BETTER."
            )

        # ── Level up ──
        if is_level_up:
            messages.append(REWARD_MSGS["level_up"])
            reward_prompts.append(
                f"You've been PROMOTED to {level_name}! "
                f"The tasks are harder now — step up your craft accordingly."
            )

        # ── Streak rewards ──
        if self._win_streak == 3:
            messages.append(REWARD_MSGS["streak_3"])
        elif self._win_streak == 5:
            messages.append(REWARD_MSGS["streak_5"])
        elif self._win_streak >= 10:
            messages.append(REWARD_MSGS["streak_10"])

        # ── Near GOTY ──
        if score >= 85:
            messages.append(REWARD_MSGS["near_goty"])
            reward_prompts.append(
                "You are 85+/100 — GOTY territory starts at 88. "
                "Push your depth_illusion and detail_density to the absolute limit."
            )

        # ── Regression penalty ──
        if is_regress:
            messages.append(PENALTY_MSGS["regress"])
            penalty_hints.append(
                f"You REGRESSED — dropped {self._prev_score - score:.1f} points. "
                f"Go back to what worked last time. Don't get sloppy."
            )

        # ── Specific weakness penalties ──
        if breakdown.get("depth_illusion", 100) < 20:
            messages.append(PENALTY_MSGS["no_depth"])
            penalty_hints.append(
                "CRITICAL: depth_illusion is near zero. "
                "You MUST use all 3 layers (background, midground, foreground). "
                "Background = desaturated/blueish. Foreground = saturated/dark at base."
            )

        if breakdown.get("coverage", 100) < 30:
            messages.append(PENALTY_MSGS["no_cover"])
            penalty_hints.append(
                "CRITICAL: You left most of the canvas empty. "
                "Fill every region — use flood_fill, rect, and gradient commands. "
                "An empty canvas gets zero composition score."
            )

        if breakdown.get("color_discipline", 100) < 30:
            messages.append(PENALTY_MSGS["no_color"])
            penalty_hints.append(
                "CRITICAL: Too few colors used. "
                "A professional pixel artist uses 8-16 colors minimum. "
                "Plan your palette: base color + shadow (×0.6) + highlight (+40% white) for each element."
            )

        if is_penalty and not penalty_hints:
            messages.append(PENALTY_MSGS["bad"] if score >= 40 else PENALTY_MSGS["terrible"])
            penalty_hints.append(
                f"Score {score:.1f} is unacceptable. "
                f"Weakest area: {min(breakdown, key=lambda k: breakdown[k] if k != 'overall' else 999)} "
                f"({min(v for k,v in breakdown.items() if k != 'overall'):.0f}/100). Fix this first."
            )

        # ── Build the penalty hint injected into next prompt ──
        combined_penalty = " | ".join(penalty_hints) if penalty_hints else ""

        # ── Build reward/motivational prompt injection ──
        combined_reward = " ".join(reward_prompts) if reward_prompts else ""
        if is_reward and not reward_prompts:
            grade, title, advice = self.get_rank(score)
            combined_reward = (
                f"Good work scoring {score:.1f}/100 ({grade} — {title}). "
                f"Next step: {advice} Keep the momentum."
            )

        # ── Determine label line ──
        if score >= 90:
            label = f"🏆 MASTERPIECE {score:.1f}/100 — GOTY QUALITY!"
        elif is_reward:
            label = f"✅ REWARD     {score:.1f}/100 — Keep climbing!"
        elif is_penalty:
            label = f"❌ PENALTY    {score:.1f}/100 — Fix your weaknesses NOW."
        else:
            label = f"⚠️  OK         {score:.1f}/100 — Not good enough yet."

        self._prev_score = score
        self._level_before = level_num

        return {
            "label":         label,
            "messages":      messages,
            "penalty_hint":  combined_penalty,
            "reward_prompt": combined_reward,
            "is_reward":     is_reward,
            "is_penalty":    is_penalty,
            "win_streak":    self._win_streak,
            "loss_streak":   self._loss_streak,
            "personal_best": self._personal_best,
            "rank":          self.get_rank(score),
        }

    def session_summary(self, avg: float) -> str:
        grade, title, advice = self.get_rank(avg)
        return (
            f"\n  🎮 SESSION RANK: {grade} — {title}"
            f"\n  💬 Coach says: {advice}"
            f"\n  🏆 Personal best this session: {self._personal_best:.1f}/100"
            f"\n  🔥 Best win streak: {self._win_streak} episodes"
        )
