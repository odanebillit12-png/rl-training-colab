"""
Animation Scorer — evaluates animation quality across 8 dimensions.

Scores 0-100, same scale as the art/world scorers.
AI Dev uses this feedback to improve each iteration.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ── Scoring dimensions ────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "frame_smoothness":    0.20,  # how smoothly each frame transitions to next
    "anticipation":        0.12,  # windup before action present
    "follow_through":      0.12,  # overshoot/settle after action
    "silhouette_clarity":  0.15,  # character readable at every frame
    "color_consistency":   0.12,  # same palette across all frames
    "effect_polish":       0.10,  # particles, glow, trail present where needed
    "godot_readiness":     0.10,  # files present, format correct
    "timing_variety":      0.09,  # fast action frames + slow settle frames
}


def score_animation(
    frames: List[Optional[object]],   # list of PIL Images or None
    animation_name: str,
    has_particles: bool = False,
    has_shader: bool = False,
    godot_files_written: bool = False,
    pixellab_generated: bool = False,
    drawn_from_scratch: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Score an animation from 0-100.
    Returns (total_score, breakdown_dict).
    """
    scores: Dict[str, float] = {}

    if PIL_AVAILABLE and frames and any(f is not None for f in frames):
        valid = [f for f in frames if f is not None]
        scores["frame_smoothness"]   = _score_smoothness(valid)
        scores["silhouette_clarity"] = _score_silhouette(valid)
        scores["color_consistency"]  = _score_color_consistency(valid)
        scores["timing_variety"]     = _score_timing(valid, animation_name)
    else:
        # Without actual images, score based on metadata
        n = len([f for f in frames if f is not None]) if frames else 0
        base = min(60.0, 40.0 + n * 2.5)
        scores["frame_smoothness"]   = base
        scores["silhouette_clarity"] = base
        scores["color_consistency"]  = base
        scores["timing_variety"]     = base

    # Heuristic scores from metadata
    scores["anticipation"]    = _score_anticipation(animation_name, frames)
    scores["follow_through"]  = _score_follow_through(animation_name, frames)
    scores["effect_polish"]   = _score_effects(has_particles, has_shader, animation_name)
    scores["godot_readiness"] = 85.0 if godot_files_written else 30.0

    # Bonus for using both modes (versatile AI Dev)
    dual_mode_bonus = 5.0 if (pixellab_generated and drawn_from_scratch) else 0.0

    total = sum(scores[k] * SCORE_WEIGHTS[k] for k in scores) + dual_mode_bonus
    total = min(100.0, total)

    return round(total, 2), {k: round(v, 1) for k, v in scores.items()}


def _score_smoothness(frames: List) -> float:
    """Score how smoothly frames transition — low pixel delta = smooth."""
    if not PIL_AVAILABLE or len(frames) < 2:
        return 55.0
    try:
        deltas = []
        for i in range(len(frames) - 1):
            a = np.array(frames[i].convert("RGBA"), dtype=float)
            b = np.array(frames[i + 1].convert("RGBA"), dtype=float)
            delta = np.mean(np.abs(a - b))
            deltas.append(delta)
        avg_delta = sum(deltas) / len(deltas)
        # Perfect: avg_delta ~20 (smooth motion), Bad: >80 (too choppy)
        score = max(0.0, 100.0 - (avg_delta - 20.0) * 1.2)
        return min(100.0, score)
    except Exception:
        return 55.0


def _score_silhouette(frames: List) -> float:
    """Score silhouette — does the character maintain readable shape?"""
    if not PIL_AVAILABLE or not frames:
        return 55.0
    try:
        areas = []
        for frame in frames:
            arr = np.array(frame.convert("RGBA"))
            alpha = arr[:, :, 3]
            areas.append(float(np.sum(alpha > 10)))
        if not areas or max(areas) == 0:
            return 30.0
        variance = float(np.std(areas) / (np.mean(areas) + 1e-6))
        # Low variance = consistent size (good), high = wildly changing (bad)
        score = max(0.0, 100.0 - variance * 80.0)
        return min(100.0, score)
    except Exception:
        return 55.0


def _score_color_consistency(frames: List) -> float:
    """Score palette consistency — same colors across all frames."""
    if not PIL_AVAILABLE or len(frames) < 2:
        return 60.0
    try:
        palettes = []
        for frame in frames:
            img = frame.convert("P", palette=Image.ADAPTIVE, colors=16)
            palette_data = img.getpalette()
            if palette_data:
                palettes.append(set(
                    (palette_data[i], palette_data[i+1], palette_data[i+2])
                    for i in range(0, 48, 3)
                ))
        if len(palettes) < 2:
            return 60.0
        base = palettes[0]
        overlaps = [len(base & p) / max(len(base | p), 1) for p in palettes[1:]]
        avg_overlap = sum(overlaps) / len(overlaps)
        return round(avg_overlap * 100.0, 1)
    except Exception:
        return 60.0


def _score_timing(frames: List, animation_name: str) -> float:
    """Score timing variety — some frames fast, some slow = good animation."""
    n = len(frames)
    if n < 4:
        return 40.0
    # More frames = better timing variety potential
    base = min(75.0, 40.0 + n * 4.0)
    # Bonus for action animations that need timing contrast
    action_keywords = ["attack", "punch", "kick", "jump", "fireball", "cast", "death"]
    if any(kw in animation_name.lower() for kw in action_keywords):
        base = min(100.0, base + 10.0)
    return base


def _score_anticipation(animation_name: str, frames: List) -> float:
    """Score whether anticipation frames exist (estimated from frame count)."""
    n = len([f for f in frames if f is not None]) if frames else 0
    action_keywords = ["attack", "punch", "kick", "jump", "cast", "fireball", "throw"]
    if not any(kw in animation_name.lower() for kw in action_keywords):
        return 70.0  # Idle/walk don't need anticipation
    if n >= 8:
        return 80.0  # Enough frames to have anticipation
    elif n >= 6:
        return 65.0
    elif n >= 4:
        return 50.0
    return 35.0


def _score_follow_through(animation_name: str, frames: List) -> float:
    """Score follow-through — estimated from frame count."""
    n = len([f for f in frames if f is not None]) if frames else 0
    if n >= 10:
        return 85.0
    elif n >= 8:
        return 70.0
    elif n >= 6:
        return 55.0
    return 40.0


def _score_effects(has_particles: bool, has_shader: bool, animation_name: str) -> float:
    """Score visual polish effects."""
    score = 30.0
    if has_particles:
        score += 35.0
    if has_shader:
        score += 25.0
    # Environmental animations don't need particles
    env_keywords = ["idle", "walk", "run", "water", "fire", "leaf"]
    if any(kw in animation_name.lower() for kw in env_keywords):
        score = max(score, 60.0)
    return min(100.0, score)


# ── Feedback generator ────────────────────────────────────────────────────────

def generate_anim_feedback(score: float, breakdown: Dict[str, float], animation_name: str) -> str:
    """Generate detailed feedback for AI Dev to improve from."""
    lines = [f"Animation '{animation_name}' scored {score:.1f}/100\n"]

    weakest = sorted(breakdown.items(), key=lambda x: x[1])[:3]
    strongest = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:2]

    lines.append("💪 STRENGTHS:")
    for dim, val in strongest:
        lines.append(f"  ✅ {dim.replace('_', ' ').title()}: {val:.0f}/100")

    lines.append("\n🔧 IMPROVE:")
    for dim, val in weakest:
        lines.append(f"  ❌ {dim.replace('_', ' ').title()}: {val:.0f}/100")
        lines.append(f"     → {_get_improvement_tip(dim, val)}")

    if score >= 90:
        lines.append("\n🌟 DIVINE QUALITY — Isekai Chronicles animation standard achieved!")
    elif score >= 80:
        lines.append("\n🏆 GOTY-level animation — almost perfect!")
    elif score >= 70:
        lines.append("\n⭐ Solid animation — polish the weak spots above.")
    elif score >= 55:
        lines.append("\n📈 Improving — focus on the red items above.")
    else:
        lines.append("\n🔰 Needs major work — re-study animation principles.")

    return "\n".join(lines)


def _get_improvement_tip(dimension: str, score: float) -> str:
    tips = {
        "frame_smoothness":   "Reduce pixel changes between consecutive frames. Ease in/out.",
        "anticipation":       "Add 1-2 windup frames BEFORE the main action. Pull back before punching.",
        "follow_through":     "Add 2-3 overshoot frames AFTER peak action. Let it settle gradually.",
        "silhouette_clarity": "Ensure the character outline is readable even at thumbnail size.",
        "color_consistency":  "Lock your palette before drawing. Same 16 colors across ALL frames.",
        "effect_polish":      "Add GPUParticles2D nodes. Use hit flash shader on impact frames.",
        "godot_readiness":    "Write SpriteFrames resource. Configure AnimationPlayer tracks.",
        "timing_variety":     "Fast frames for action peak, slow frames for start/end of motion.",
    }
    return tips.get(dimension, "Study reference animations and apply the 12 principles.")
