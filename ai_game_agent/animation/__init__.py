"""
AI Dev Animation Module — PixelLab generation + draw-from-scratch + Godot assembly.

AI Dev learns to:
  1. Generate great animations via PixelLab API (template + custom modes)
  2. Draw animation frames from scratch pixel-by-pixel (no API needed)
  3. Combine frames, polish effects, and assemble in Godot
  4. Score its own work and improve over training episodes
"""
from .animation_curriculum import ANIMATION_CURRICULUM, get_anim_level, get_next_anim_task
from .animation_scorer import score_animation, generate_anim_feedback

__all__ = [
    "ANIMATION_CURRICULUM",
    "get_anim_level",
    "get_next_anim_task",
    "score_animation",
    "generate_anim_feedback",
]
