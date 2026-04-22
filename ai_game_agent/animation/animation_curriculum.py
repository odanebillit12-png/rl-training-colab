"""
Animation Curriculum — 10 levels from basic walk cycle to DIVINE-tier polished characters.

Each level defines:
  - task_name     : what to animate
  - frame_count   : how many frames to draw
  - directions    : which directions to cover
  - mode          : "pixellab" | "draw" | "both"  (both = do both, take the better)
  - graduate_at   : score needed to advance
  - bonus_tasks   : optional extra polish requirements
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

PIXELLAB_TEMPLATES = [
    "walk", "walking", "walking-4-frames", "walking-6-frames", "walking-8-frames",
    "running-4-frames", "running-6-frames", "running-8-frames",
    "jumping-1", "jumping-2", "running-jump",
    "fight-stance-idle-8-frames", "breathing-idle",
    "lead-jab", "cross-punch", "roundhouse-kick", "high-kick",
    "flying-kick", "hurricane-kick", "leg-sweep",
    "fireball", "throw-object", "taking-punch",
    "crouching", "crouched-walking",
    "front-flip", "backflip", "running-slide",
    "falling-back-death", "getting-up",
    "picking-up", "drinking", "pushing", "pull-heavy-object",
    "sad-walk", "scary-walk", "surprise-uppercut", "two-footed-jump",
]

@dataclass
class AnimLevel:
    level: int
    name: str
    task_name: str
    description: str
    frame_count: int
    directions: List[str]
    mode: str                  # "pixellab" | "draw" | "both"
    template_id: Optional[str] # which PixelLab template to use (if mode includes pixellab)
    graduate_at: float
    size: int = 48             # sprite canvas size
    bonus_tasks: List[str] = field(default_factory=list)
    custom_prompt: Optional[str] = None  # for PixelLab custom animation

ANIMATION_CURRICULUM: List[AnimLevel] = [
    AnimLevel(
        level=1,
        name="First Steps",
        task_name="4-frame walk south",
        description="Basic walk cycle facing south — 4 frames, simple character.",
        frame_count=4,
        directions=["south"],
        mode="both",
        template_id="walking-4-frames",
        graduate_at=55.0,
        bonus_tasks=["frames are evenly spaced", "foot contacts visible"],
    ),
    AnimLevel(
        level=2,
        name="Idle Breathing",
        task_name="idle breathing animation",
        description="Subtle breathing idle loop — chest rises/falls, 4-8 frames.",
        frame_count=6,
        directions=["south"],
        mode="both",
        template_id="breathing-idle",
        graduate_at=60.0,
        bonus_tasks=["loop is seamless", "head bobs slightly"],
    ),
    AnimLevel(
        level=3,
        name="All Directions Walk",
        task_name="8-direction walk cycle",
        description="Full 8-direction walk — south/north/east/west + diagonals.",
        frame_count=6,
        directions=["south", "north", "east", "west", "south-east", "south-west", "north-east", "north-west"],
        mode="pixellab",
        template_id="walking-6-frames",
        graduate_at=65.0,
        bonus_tasks=["character silhouette consistent across directions"],
    ),
    AnimLevel(
        level=4,
        name="Run Cycle",
        task_name="8-frame run cycle",
        description="Fast run with clear anticipation lean and full stride.",
        frame_count=8,
        directions=["south", "east", "west", "north"],
        mode="both",
        template_id="running-8-frames",
        graduate_at=68.0,
        bonus_tasks=["hair/cape moves with momentum", "dust kick on foot contact"],
    ),
    AnimLevel(
        level=5,
        name="Combat Basics",
        task_name="sword attack with anticipation",
        description="Attack animation: windup → slash → follow-through → recovery.",
        frame_count=8,
        directions=["south", "east"],
        mode="both",
        template_id="cross-punch",
        graduate_at=72.0,
        custom_prompt="quick precise sword slash with windup and follow-through",
        bonus_tasks=["anticipation frame before slash", "follow-through overshoot", "recovery settle"],
    ),
    AnimLevel(
        level=6,
        name="Hit & Death",
        task_name="hit reaction and death fall",
        description="Taking a hit (knockback stagger) + death fall animation.",
        frame_count=8,
        directions=["south"],
        mode="both",
        template_id="taking-punch",
        graduate_at=75.0,
        bonus_tasks=["squash on impact", "death has final settling frame"],
    ),
    AnimLevel(
        level=7,
        name="Magic Effects",
        task_name="fireball cast with particle trail",
        description="Spellcast: charging glow → release → projectile arc with trail particles.",
        frame_count=10,
        directions=["south", "east"],
        mode="both",
        template_id="fireball",
        custom_prompt="casting glowing fireball with charging windup and trail particles",
        graduate_at=78.0,
        bonus_tasks=["charge glow grows each frame", "trail fades alpha", "impact flash frame"],
        size=64,
    ),
    AnimLevel(
        level=8,
        name="Combat Combo",
        task_name="3-hit combo attack",
        description="Full combat combo: jab → kick → finisher with screen flash.",
        frame_count=12,
        directions=["east"],
        mode="both",
        template_id="hurricane-kick",
        custom_prompt="aggressive three-hit combat combo jab kick spinning finisher",
        graduate_at=82.0,
        bonus_tasks=["each hit has impact frame", "hitbox timing aligns", "combo flows smoothly"],
        size=64,
    ),
    AnimLevel(
        level=9,
        name="Environmental FX",
        task_name="water ripple, fire flicker, leaf drift",
        description="Environmental animations from scratch: looping water/fire/particles.",
        frame_count=8,
        directions=["south"],
        mode="draw",          # must draw these from scratch — no PixelLab template
        template_id=None,
        custom_prompt="water ripple with subtle foam and depth shimmer",
        graduate_at=88.0,
        bonus_tasks=["loop perfectly seamless", "color temperature shift mid-animation", "particle fade-out"],
        size=32,
    ),
    AnimLevel(
        level=10,
        name="DIVINE Character",
        task_name="full character animation sheet polished to perfection",
        description=(
            "Complete animated character: idle + walk + run + attack + cast + hit + death. "
            "All 8 directions. Every frame hand-polished. Effects: glow, particles, outlines. "
            "Assembled as SpriteFrames in Godot with AnimationPlayer tracks."
        ),
        frame_count=8,
        directions=["south", "north", "east", "west", "south-east", "south-west", "north-east", "north-west"],
        mode="both",
        template_id="walking-8-frames",
        custom_prompt="full hero character animation all directions walk run attack cast hit death",
        graduate_at=135.0,   # DIVINE ceiling
        bonus_tasks=[
            "all animations loop seamlessly",
            "particle effects on magic attacks",
            "squash and stretch on jumps",
            "screen shake on heavy attacks",
            "Godot SpriteFrames assembled",
            "AnimationPlayer tracks configured",
            "GPUParticles attached for spells",
        ],
        size=64,
    ),
]


def get_anim_level(score: float) -> AnimLevel:
    """Return the curriculum level appropriate for the current score."""
    level = 1
    for lv in ANIMATION_CURRICULUM:
        if score >= lv.graduate_at:
            level = min(lv.level + 1, 10)
    return ANIMATION_CURRICULUM[level - 1]


def get_next_anim_task(current_level: int, episode: int) -> AnimLevel:
    """Return the task for this level, cycling through sub-tasks."""
    idx = min(current_level - 1, len(ANIMATION_CURRICULUM) - 1)
    return ANIMATION_CURRICULUM[idx]
