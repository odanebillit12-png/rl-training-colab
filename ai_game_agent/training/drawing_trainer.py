"""
Drawing Trainer — AI Dev's complete art school training loop.

AI Dev:
  1. STUDIES reference pixel art + anime images (Gemini Vision analysis)
  2. DRAWS from scratch on a blank canvas using pencil commands
  3. Gets SCORED on 8 quality criteria
  4. Receives PENALTY (bad feedback) or REWARD (praise + level up)
  5. Improves over hundreds of episodes until GOTY-level quality

Run:
    python3 -m ai_game_agent.training.drawing_trainer

Or import and call:
    from ai_game_agent.training.drawing_trainer import DrawingTrainer, DrawingConfig
    trainer = DrawingTrainer(DrawingConfig(max_episodes=50))
    asyncio.run(trainer.run())
"""
from __future__ import annotations
import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..drawing.pixel_canvas import PixelCanvas
from ..drawing.reference_library import ReferenceLibrary
from ..drawing.drawing_agent import DrawingAgent, RateLimitError
from ..drawing.quality_scorer import score_drawing, generate_feedback
from ..drawing.curriculum import DRAWING_CURRICULUM, get_current_level, get_next_task
from .motivation_engine import MotivationEngine


@dataclass
class DrawingConfig:
    max_episodes: int = 100
    target_score: float = 88.0
    reward_threshold: float = 75.0
    penalty_threshold: float = 50.0
    rolling_window: int = 10
    save_all: bool = True
    save_dir: str = "training_data/drawn_art"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    # github_api_key accepts a comma-separated list of PATs for rotation
    github_api_key: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    brave_api_key: str = field(default_factory=lambda: os.getenv("BRAVE_API_KEY", ""))
    study_references_first: bool = True
    download_refs: bool = True


class DrawingTrainer:
    """AI Dev's pixel art training school — from beginner to GOTY master."""

    def __init__(self, config: DrawingConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Mirror drawings into Godot assets folder automatically
        self.godot_mirror_dir = Path("godot_ai_colony/assets/ai_dev_drawings")
        self.godot_mirror_dir.mkdir(parents=True, exist_ok=True)

        # Reference library (art school study materials)
        self.ref_library = ReferenceLibrary()

        # Drawing agent (AI Dev's brain — Claude primary, GitHub Models secondary, Groq fallback, Gemini last)
        self.agent = DrawingAgent(
            api_key=config.gemini_api_key,
            anthropic_key=config.anthropic_api_key,
            github_key=config.github_api_key,
            groq_key=config.groq_api_key,
            brave_key=config.brave_api_key,
            ref_library=self.ref_library,
        )

        # Training state
        self._scores: deque = deque(maxlen=config.rolling_window)
        self._episode_count = 0
        self._penalty_hint = ""
        self._reward_prompt = ""
        self._stats_file = Path("training_data/drawing_stats.json")
        self._motivator = MotivationEngine(
            reward_threshold=config.reward_threshold,
            penalty_threshold=config.penalty_threshold,
        )
        self._load_stats()

    # ── Stats persistence ─────────────────────────────────────────────────

    def _load_stats(self) -> None:
        if self._stats_file.exists():
            try:
                data = json.loads(self._stats_file.read_text())
                self._episode_count = data.get("total_episodes", 0)
                prev_scores = data.get("recent_scores", [])
                for s in prev_scores[-self.config.rolling_window:]:
                    self._scores.append(s)
                print(f"  📚 Resuming: {self._episode_count} prior episodes, "
                      f"avg={self.rolling_avg:.1f}")
            except Exception:
                pass

    def _save_stats(self, extra: dict | None = None) -> None:
        data = {
            "total_episodes": self._episode_count,
            "recent_scores": list(self._scores),
            "rolling_avg": self.rolling_avg,
        }
        if extra:
            data.update(extra)
        self._stats_file.parent.mkdir(parents=True, exist_ok=True)
        self._stats_file.write_text(json.dumps(data, indent=2))

    @property
    def rolling_avg(self) -> float:
        return sum(self._scores) / len(self._scores) if self._scores else 0.0

    # ── Reference study phase ─────────────────────────────────────────────

    def study_references(self) -> None:
        """
        AI Dev goes to art school first — studies pixel art and anime references.
        Uses Gemini Vision to extract techniques from each image.
        """
        print("\n" + "="*60)
        print("📚 AI DEV ART SCHOOL — STUDYING REFERENCES")
        print("="*60)

        # 1. Download static CC0 references
        if self.config.download_refs:
            print("\n📥 Downloading CC0 reference images...")
            n = self.ref_library.download_references(verbose=True)
            print(f"   Downloaded {n} new references")

        # 2. Search for pixel art images via DuckDuckGo (free, no key needed)
        searched = self.ref_library.search_pixel_art_images(verbose=True)
        if searched:
            print(f"   🔍 Found {searched} images via image search")

        # 3. Check for all references (user-placed + downloaded + searched)
        user_refs = self.ref_library.index_user_references()
        print(f"\n🖼️  Found {len(user_refs)} reference images to study:")
        for r in user_refs[:10]:
            print(f"   • {r.name}")
        if len(user_refs) > 10:
            print(f"   ... and {len(user_refs) - 10} more")

        if not user_refs:
            print("   ℹ️  No reference images found. Using built-in technique lessons.")
        else:
            # Analyze with Gemini Vision
            print(f"\n🔍 Analyzing references with Gemini Vision...")
            added = self.ref_library.analyze_all_references(self.config.gemini_api_key)
            print(f"   Learned {added} new technique sets")

        # Show what we know
        total_lessons = self.ref_library.lesson_count()
        print(f"\n✅ AI Dev has {total_lessons} technique lessons loaded:")
        for key, lesson in self.ref_library._lessons.items():
            name = lesson.get("technique", key)
            print(f"   • {name}")

        # Compile into agent's working memory
        self.agent._compile_techniques()
        print(f"\n🧠 Techniques compiled into AI Dev's drawing memory")

    # ── Main training loop ────────────────────────────────────────────────

    async def run(self) -> Dict:
        """Run the full training loop. Returns final statistics."""
        print("\n" + "="*60)
        print("🎨 AI DEV — PIXEL ART TRAINING (Draw from Scratch)")
        print(f"   Target : {self.config.target_score}/100 (GOTY Level)")
        print(f"   Episodes: {self.config.max_episodes}")
        print(f"   Output  : {self.save_dir}")
        print("="*60)

        # Phase 1: Study references
        if self.config.study_references_first:
            self.study_references()

        print("\n\n" + "="*60)
        print("🖊️  DRAWING TRAINING — AI Dev picks up the pencil")
        print("="*60 + "\n")

        new_episodes = 0
        best_score = max(self._scores, default=0.0)

        for ep in range(self.config.max_episodes):
            self._episode_count += 1
            new_episodes += 1

            # Get curriculum level
            level = get_current_level(self.rolling_avg)
            task = get_next_task(level, ep)

            w, h = task.get("width", 64), task.get("height", 64)
            print(f"\n── Episode {self._episode_count} [Level {level['level']}: {level['name']}] ──")
            print(f"   Task    : {task['task_name']} ({w}×{h})")
            print(f"   Subject : {task['description'][:70]}...")
            print(f"   Rolling avg: {self.rolling_avg:.1f}/100")

            # Build save path
            img_path = str(
                self.save_dir /
                f"ep{self._episode_count:04d}_L{level['level']}_{task['task_name']}.png"
            )

            # ── DRAW ──
            t0 = time.time()
            try:
                canvas, commands, plan = self.agent.full_draw_session(
                    task=task,
                    penalty_hint=self._penalty_hint,
                    reward_prompt=self._reward_prompt,
                    save_path=img_path if self.config.save_all else None,
                )
            except RateLimitError as rle:
                # All APIs rate-limited — progressive backoff, cap at 5 min
                self._episode_count -= 1
                new_episodes -= 1
                rate_limit_hits = getattr(self, '_rate_limit_hits', 0) + 1
                self._rate_limit_hits = rate_limit_hits
                cooldown = min(60 * rate_limit_hits, 300)
                print(f"\n   ⏸  All APIs rate-limited (hit #{rate_limit_hits})")
                print(f"   ⏸  Waiting {cooldown}s for quotas to reset...")
                time.sleep(cooldown)
                continue
            self._rate_limit_hits = 0  # reset on successful call
            draw_time = time.time() - t0

            # ── SCORE ──
            final_img = canvas.composite()
            breakdown = score_drawing(final_img, task)
            score = breakdown["overall"]
            feedback = generate_feedback(breakdown, task)

            print(f"   ⏱️  Drew in {draw_time:.1f}s | {canvas.stroke_count()} strokes | "
                  f"{len(canvas.unique_colors())} colors used")
            print(f"   📊 Scores:")
            for k, v in breakdown.items():
                if k != "overall":
                    bar = "█" * int(v / 10) + "░" * (10 - int(v / 10))
                    print(f"      {k:20s} {bar} {v:.0f}")
            print(f"   {'='*50}")

            # ── REWARD / PENALTY (Motivation Engine) ──
            self._scores.append(score)
            if score > best_score:
                best_score = score
                best_path = str(self.save_dir / "BEST_SO_FAR.png")
                final_img.save(best_path)
                # Mirror best into Godot
                import shutil
                shutil.copy2(best_path, str(self.godot_mirror_dir / "BEST_SO_FAR.png"))

            # Mirror composite drawing into Godot assets automatically
            if self.config.save_all and img_path:
                import shutil
                godot_dest = self.godot_mirror_dir / Path(img_path).name
                if not godot_dest.exists():
                    shutil.copy2(img_path, str(godot_dest))

            motivation = self._motivator.evaluate(
                score=score,
                breakdown=breakdown,
                episode=self._episode_count,
                level_name=level["name"],
                level_num=level["level"],
            )

            print(f"   {'='*50}")
            print(f"   {motivation['label']}")
            for msg in motivation["messages"]:
                print(f"   {msg}")

            grade, title, advice = motivation["rank"]
            print(f"   🎖  Rank: {grade} — {title}")
            if motivation["win_streak"] >= 2:
                print(f"   🔥 Win streak: {motivation['win_streak']} episodes!")
            if motivation["loss_streak"] >= 2:
                print(f"   💀 Loss streak: {motivation['loss_streak']} — fix something NOW.")

            # Inject feedback into agent's next draw
            self._penalty_hint  = motivation["penalty_hint"]
            self._reward_prompt = motivation["reward_prompt"]

            # Self-critique for learning
            critique = self.agent.self_critique(score, breakdown, task)
            print(f"   🤔 AI Dev: {critique[:120]}")

            # Record episode in memory (convert numpy types for JSON serialization)
            clean_breakdown = {k: float(v) for k, v in breakdown.items()}
            self.agent.add_episode({
                "episode": self._episode_count,
                "task": task["task_name"],
                "level": level["level"],
                "score": float(score),
                "breakdown": clean_breakdown,
                "feedback": feedback,
                "critique": critique,
                "strokes": canvas.stroke_count(),
                "colors": len(canvas.unique_colors()),
                "saved_to": img_path if self.config.save_all else "",
            })

            # Save stats
            self._save_stats({
                "best_score": best_score,
                "current_level": level["level"],
                "current_level_name": level["name"],
            })

            # Check if target reached
            if self.rolling_avg >= self.config.target_score and len(self._scores) >= 5:
                print(f"\n🎉 TARGET REACHED! Rolling avg {self.rolling_avg:.1f} ≥ {self.config.target_score}")
                print(f"   AI Dev has achieved GOTY-level pixel art!")
                break

            # Brief pause between episodes
            await asyncio.sleep(2)

        # ── Final summary ──
        stats = {
            "total_episodes": self._episode_count,
            "new_this_run": new_episodes,
            "avg_score": round(float(self.rolling_avg), 1),
            "best_score": round(float(best_score), 1),
            "final_level": get_current_level(self.rolling_avg)["level"],
            "final_level_name": get_current_level(self.rolling_avg)["name"],
            "output_dir": str(self.save_dir),
        }

        print(f"\n{'='*60}")
        print(f"🎨 TRAINING COMPLETE")
        print(f"   Episodes   : {new_episodes} new  ({self._episode_count} total)")
        print(f"   Avg Score  : {self.rolling_avg:.1f}/100")
        print(f"   Best Score : {best_score:.1f}/100")
        print(f"   Level      : {stats['final_level']} — {stats['final_level_name']}")
        print(f"   Art saved  : {self.save_dir}/")
        print(self._motivator.session_summary(self.rolling_avg))
        print(f"{'='*60}\n")
        return stats


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Dev Pixel Art Training")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--target", type=float, default=88.0)
    parser.add_argument("--no-refs", action="store_true", help="Skip reference study")
    parser.add_argument("--no-download", action="store_true", help="Don't download references")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        exit(1)

    cfg = DrawingConfig(
        max_episodes=args.episodes,
        target_score=args.target,
        gemini_api_key=api_key,
        study_references_first=not args.no_refs,
        download_refs=not args.no_download,
    )
    trainer = DrawingTrainer(cfg)
    asyncio.run(trainer.run())
