"""
Animation Trainer — Phase 3 of AI Dev's training pipeline.

AI Dev trains to create DIVINE-tier animations by:

  MODE A — PixelLab Generation:
    1. Study what makes great animations (animation principles)
    2. Create/reuse a character via PixelLab API
    3. Generate template animation (1 gen, cheap!)
    4. If no template exists, draw from scratch instead
    5. Download frames, score quality

  MODE B — Draw From Scratch:
    1. Study animation principles (squash/stretch, anticipation, etc.)
    2. Draw each frame individually using pixel commands
    3. Compose into spritesheet
    4. Score quality + compare to PixelLab result

  BOTH MODES RUN IN PARALLEL when time allows.
  Best result is kept. AI Dev learns from both.

  GODOT ASSEMBLY:
    - Writes SpriteFrames GDScript
    - Creates AnimationPlayer tracks
    - Attaches particle effects
    - Applies hit flash / outline shaders
    - Saves to godot_ai_colony/assets/characters/

Run standalone:
    python3 -m ai_game_agent.training.animation_trainer
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..animation.animation_curriculum import (
    ANIMATION_CURRICULUM, AnimLevel, get_anim_level, get_next_anim_task,
    PIXELLAB_TEMPLATES,
)
from ..animation.animation_scorer import score_animation, generate_anim_feedback
from ..animation.frame_painter import FramePainter
from ..animation.godot_animator import GodotAnimator
from ..animation.pixellab_animator import PixelLabAnimator


@dataclass
class AnimationConfig:
    max_episodes: int = 50
    target_score: float = 75.0
    rolling_window: int = 10
    save_dir: str = "training_data/animations"
    asset_dir: str = "godot_ai_colony/assets/characters"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    pixellab_api_key: str = field(default_factory=lambda: os.getenv("PIXELLAB_API_KEY", ""))
    # Reuse an existing character rather than creating new ones every run
    reuse_character_id: str = ""


class AnimationTrainer:
    """AI Dev's animation mastery training loop."""

    def __init__(self, config: AnimationConfig):
        self.config = config
        self.on_episode: Optional[Callable] = None

        self.painter = FramePainter(
            api_key=config.gemini_api_key or config.groq_api_key,
            model="gemini" if config.gemini_api_key else "groq",
        )
        self.pixellab = PixelLabAnimator(api_key=config.pixellab_api_key)
        self.godot = GodotAnimator()

        self._save_dir = Path(config.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._scores: deque = deque(maxlen=config.rolling_window)
        self._best_score: float = 0.0
        self._curriculum_level: int = 1

    async def run(self) -> dict:
        """Run training loop and return final stats."""
        stats: Dict = {
            "episodes": 0,
            "best_score": 0.0,
            "avg_score": 0.0,
            "curriculum_level": 1,
            "animations_created": [],
        }

        for ep in range(self.config.max_episodes):
            task = get_next_anim_task(self._curriculum_level, ep)
            score, info = await self._run_episode(ep + 1, task)

            self._scores.append(score)
            avg = sum(self._scores) / len(self._scores)
            reward = score >= task.graduate_at * 0.8
            penalty = score < task.graduate_at * 0.4

            if score > self._best_score:
                self._best_score = score

            # Advance curriculum level
            if avg >= task.graduate_at and self._curriculum_level < 10:
                self._curriculum_level += 1

            stats["episodes"] += 1
            stats["best_score"] = self._best_score
            stats["avg_score"] = avg
            stats["curriculum_level"] = self._curriculum_level
            if info.get("saved_path"):
                stats["animations_created"].append(info["saved_path"])

            if self.on_episode:
                self.on_episode({
                    "episode": ep + 1,
                    "score": score,
                    "rolling_avg": avg,
                    "reward": reward,
                    "penalty": penalty,
                    "level": self._curriculum_level,
                    "task_name": task.task_name,
                    "mode": info.get("mode_used", "?"),
                    "godot_ready": info.get("godot_ready", False),
                })

        return stats

    async def _run_episode(self, ep: int, task: AnimLevel) -> tuple[float, dict]:
        """Run one animation episode. Try both modes, take best."""
        info: Dict = {"mode_used": task.mode, "godot_ready": False}
        pixellab_score = 0.0
        scratch_score = 0.0
        best_frames = []

        # ── MODE A: PixelLab generation ──────────────────────────────────────
        if task.mode in ("pixellab", "both") and self.pixellab.is_available():
            pl_score, pl_frames, pl_path = await self._run_pixellab_episode(task, ep)
            pixellab_score = pl_score
            if pl_score > scratch_score:
                best_frames = pl_frames
                info["saved_path"] = str(pl_path) if pl_path else ""
                info["mode_used"] = "pixellab"

        # ── MODE B: Draw from scratch ─────────────────────────────────────────
        if task.mode in ("draw", "both"):
            dr_score, dr_frames, dr_path = await self._run_draw_episode(task, ep)
            scratch_score = dr_score
            if dr_score > pixellab_score:
                best_frames = dr_frames
                info["saved_path"] = str(dr_path) if dr_path else ""
                info["mode_used"] = "draw_scratch"

        final_score = max(pixellab_score, scratch_score)

        # ── GODOT ASSEMBLY (if we have frames and score is decent) ────────────
        if best_frames and final_score >= 50.0:
            godot_ready = await self._assemble_in_godot(task, best_frames, ep)
            info["godot_ready"] = godot_ready

        # Generate feedback for learning
        if final_score < self.config.target_score:
            _, breakdown = score_animation(
                best_frames, task.task_name,
                has_particles=info.get("godot_ready", False),
                has_shader=info.get("godot_ready", False),
            )
            feedback = generate_anim_feedback(final_score, breakdown, task.task_name)
            self._save_feedback(ep, task.task_name, final_score, feedback)

        return round(final_score, 2), info

    async def _run_pixellab_episode(
        self, task: AnimLevel, ep: int
    ) -> tuple[float, list, Optional[Path]]:
        """Generate animation via PixelLab API and score it."""
        try:
            # Get or create character
            char_id = self.config.reuse_character_id
            if not char_id:
                char_id = await self._get_or_create_character(task)
            if not char_id:
                return 0.0, [], None

            # Use template if available (cheap!), fall back to draw
            if task.template_id and task.template_id in PIXELLAB_TEMPLATES:
                job_ids = self.pixellab.animate_template(
                    char_id,
                    task.template_id,
                    directions=task.directions[:2],  # limit to 2 dirs in training
                    animation_name=task.task_name,
                )
            elif task.custom_prompt:
                # Custom is expensive — only use in training if cost confirmed
                job_ids, cost = self.pixellab.animate_custom(
                    char_id,
                    action_description=task.custom_prompt,
                    directions=task.directions[:1],  # south only for training
                    confirmed=False,  # never auto-confirm expensive ops
                )
                if not job_ids:
                    # Too expensive — switch to draw mode
                    return 0.0, [], None
            else:
                return 0.0, [], None

            # Wait for completion
            anim_data = self.pixellab.wait_for_animation(char_id, max_wait=180)
            if not anim_data:
                return 0.0, [], None

            # Download frames
            ep_dir = self._save_dir / f"ep{ep:04d}_pixellab"
            saved_paths = self.pixellab.download_animation_frames(
                anim_data, ep_dir, task.task_name
            )

            # Load frames for scoring
            frames = self._load_frame_images(saved_paths)
            score, _ = score_animation(
                frames, task.task_name,
                pixellab_generated=True,
                godot_files_written=False,
            )
            return score, frames, saved_paths[0] if saved_paths else None

        except Exception as e:
            print(f"    ⚠️  PixelLab episode error: {e}")
            return 0.0, [], None

    async def _run_draw_episode(
        self, task: AnimLevel, ep: int
    ) -> tuple[float, list, Optional[Path]]:
        """Draw animation frames from scratch and score them."""
        try:
            ep_dir = self._save_dir / f"ep{ep:04d}_scratch"
            self.painter.width = task.size
            self.painter.height = task.size

            principles = self._get_animation_principles(task.task_name)

            saved_path = await self.painter.draw_animation(
                task_name=task.task_name,
                description=task.description,
                frame_count=task.frame_count,
                animation_principles=principles,
                save_dir=ep_dir,
            )

            frames = []
            if saved_path and Path(saved_path).exists():
                frames = self._split_spritesheet(saved_path, task.size, task.frame_count)

            score, _ = score_animation(
                frames, task.task_name,
                drawn_from_scratch=True,
                godot_files_written=False,
            )
            return score, frames, saved_path

        except Exception as e:
            print(f"    ⚠️  Draw episode error: {e}")
            return 0.0, [], None

    async def _assemble_in_godot(
        self, task: AnimLevel, frames: list, ep: int
    ) -> bool:
        """Assemble animation frames into Godot SpriteFrames + controller."""
        try:
            frame_paths: List[Path] = []
            ep_dir = self._save_dir / f"ep{ep:04d}_frames"
            ep_dir.mkdir(parents=True, exist_ok=True)

            for i, frame in enumerate(frames):
                if frame is not None:
                    p = ep_dir / f"frame_{i:02d}.png"
                    frame.save(str(p))
                    frame_paths.append(p)

            if not frame_paths:
                return False

            clean_name = task.task_name.replace(" ", "_").lower()
            result = self.godot.build_animated_sprite(
                character_name=clean_name,
                animation_frames={task.task_name.split()[0]: frame_paths},
                fps=8,
                output_dir=self.config.asset_dir,
            )

            # Save Godot setup instructions for later MCP execution
            instructions_path = self._save_dir / f"ep{ep:04d}_godot_instructions.json"
            instructions_path.write_text(json.dumps(result, indent=2, default=str))

            return bool(result.get("godot_setup"))

        except Exception as e:
            print(f"    ⚠️  Godot assembly error: {e}")
            return False

    async def _get_or_create_character(self, task: AnimLevel) -> Optional[str]:
        """Get existing character ID or create a new training character."""
        # Check if we have a saved training character
        char_cache = self._save_dir / "training_character.json"
        if char_cache.exists():
            try:
                data = json.loads(char_cache.read_text())
                return data.get("character_id")
            except Exception:
                pass

        # Create a new training character
        char_id = self.pixellab.create_character(
            description="hero warrior with dark armor and glowing blue eyes",
            name="Training Hero",
            mode="standard",
            n_directions=4,  # 4 directions only in training (cheaper)
            size=task.size,
            view="low top-down",
        )
        if not char_id:
            return None

        # Wait for character to be ready
        char_data = self.pixellab.get_character(char_id, max_wait=240)
        if not char_data:
            return None

        char_cache.write_text(json.dumps({"character_id": char_id, "data": char_data}, default=str))
        return char_id

    def _get_animation_principles(self, task_name: str) -> List[str]:
        """Return the most relevant animation principles for this task."""
        task_lower = task_name.lower()
        if "walk" in task_lower or "run" in task_lower:
            return ["secondary action", "ease in/out", "arcs", "overlap"]
        elif "attack" in task_lower or "punch" in task_lower or "kick" in task_lower:
            return ["anticipation", "follow-through", "squash and stretch", "timing"]
        elif "jump" in task_lower or "flip" in task_lower:
            return ["squash and stretch", "anticipation", "follow-through", "arcs"]
        elif "idle" in task_lower or "breath" in task_lower:
            return ["ease in/out", "secondary action", "timing"]
        elif "death" in task_lower or "hit" in task_lower or "fall" in task_lower:
            return ["squash and stretch", "follow-through", "timing"]
        elif "fire" in task_lower or "cast" in task_lower or "spell" in task_lower:
            return ["anticipation", "follow-through", "secondary action", "timing"]
        elif "water" in task_lower or "fire" in task_lower or "leaf" in task_lower:
            return ["ease in/out", "secondary action", "overlap"]
        return ["ease in/out", "timing", "secondary action"]

    def _load_frame_images(self, paths: List[Path]) -> list:
        """Load PIL Images from file paths."""
        frames = []
        if not PIL_AVAILABLE_CHECK:
            return frames
        try:
            from PIL import Image
            for p in paths:
                p = Path(p)
                if p.exists():
                    frames.append(Image.open(str(p)).copy())
        except Exception:
            pass
        return frames

    def _split_spritesheet(self, sheet_path, frame_size: int, n_frames: int) -> list:
        """Split a horizontal spritesheet into individual PIL Image frames."""
        if not PIL_AVAILABLE_CHECK:
            return []
        try:
            from PIL import Image
            sheet = Image.open(str(sheet_path))
            frames = []
            for i in range(n_frames):
                x = i * frame_size
                if x + frame_size <= sheet.width:
                    frame = sheet.crop((x, 0, x + frame_size, frame_size))
                    frames.append(frame)
            return frames
        except Exception:
            return []

    def _save_feedback(self, ep: int, task: str, score: float, feedback: str):
        log_file = self._save_dir / "animation_training.log"
        ts = time.strftime("%H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"\n[{ts}] Ep{ep:04d} | {task} | Score: {score:.1f}\n")
            f.write(feedback + "\n")


# Check PIL availability at module level
try:
    from PIL import Image  # noqa: F401
    PIL_AVAILABLE_CHECK = True
except ImportError:
    PIL_AVAILABLE_CHECK = False


if __name__ == "__main__":
    async def _main():
        cfg = AnimationConfig(max_episodes=5)

        def on_ep(info):
            icon = "✅" if info["reward"] else ("❌" if info["penalty"] else "⚠️")
            print(f"  {icon} Anim Ep{info['episode']:3d} L{info['level']} "
                  f"Score:{info['score']:5.1f} Avg:{info['rolling_avg']:5.1f} "
                  f"Mode:{info['mode']} Godot:{info['godot_ready']} | {info['task_name']}")

        trainer = AnimationTrainer(cfg)
        trainer.on_episode = on_ep
        stats = await trainer.run()
        print(f"\nFinal: best={stats['best_score']:.1f} level={stats['curriculum_level']}")

    asyncio.run(_main())
