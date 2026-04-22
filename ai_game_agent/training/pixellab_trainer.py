"""
PixelLab Curriculum Trainer
============================
AI Dev learns to draw pixel art by generating images through the PixelLab API,
evaluating their quality, and iteratively improving its prompts.

Curriculum (from basic to expert):
  Level 1 — First Pixels  : 16×16 tiles, simple descriptions
  Level 2 — Tile Mastery  : 32×32 tiles, wang tilesets, sidescroller platforms
  Level 3 — Characters    : 32×32 humanoid characters (standard mode)
  Level 4 — Animation     : Animated characters, 8 directions, walk cycles
  Level 5 — Expert        : Pro-mode characters, full asset packs, isometric maps

The trainer calls the PixelLab REST API directly so it can run anywhere —
locally, in Google Colab, or in a GitHub Actions runner.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import requests

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Config ─────────────────────────────────────────────────────────────────────

PIXELLAB_BASE_V2 = "https://api.pixellab.ai/v2"


@dataclass
class PixelLabConfig:
    api_key: str = ""                        # set from env PIXELLAB_API_KEY
    max_episodes: int = 100
    target_score: float = 80.0
    penalty_threshold: float = 40.0
    reward_threshold: float = 65.0
    window_size: int = 10
    save_images: bool = True
    output_dir: str = "training_data/pixellab_art"

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("PIXELLAB_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "PixelLab API key required. Set PIXELLAB_API_KEY environment variable "
                "or pass api_key= to PixelLabConfig."
            )


# ── Curriculum ─────────────────────────────────────────────────────────────────

PIXELLAB_CURRICULUM = [
    # ── Level 1: First Pixels ──────────────────────────────────────────────
    {
        "level": 1,
        "name": "First Pixels",
        "min_score": 0,
        "description": "Learn to generate clean, simple 16×16 tiles. Master basic descriptions.",
        "tasks": [
            {
                "action": "create_tile",
                "task_name": "grass_tile",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "bright green grass ground tile, flat pixel art, simple, clean",
                    "image_size": {"width": 16, "height": 16},
                },
                "target_description": "16×16 grass tile — flat colours, 4–8 colours",
            },
            {
                "action": "create_tile",
                "task_name": "dirt_tile",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "brown earthy dirt ground tile, flat pixel art, simple",
                    "image_size": {"width": 16, "height": 16},
                },
                "target_description": "16×16 dirt tile — earthy browns, 4–6 colours",
            },
            {
                "action": "create_tile",
                "task_name": "stone_tile",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "grey stone floor tile, pixel art, subtle shading",
                    "image_size": {"width": 16, "height": 16},
                },
                "target_description": "16×16 stone tile — grey palette, some shading",
            },
            {
                "action": "create_tile",
                "task_name": "water_tile",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "blue water tile, pixel art, simple ripple pattern",
                    "image_size": {"width": 16, "height": 16},
                },
                "target_description": "16×16 water tile — blues, ripple pattern",
            },
        ],
    },

    # ── Level 2: Tile Mastery ──────────────────────────────────────────────
    {
        "level": 2,
        "name": "Tile Mastery",
        "min_score": 40,
        "description": "Generate 32×32 tiles with colour detail and texture.",
        "tasks": [
            {
                "action": "create_tile",
                "task_name": "grass_32",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "lush green grass ground tile, pixel art, detailed grass blades, top-down RPG",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 grass tile — medium detail, 8–12 colours",
            },
            {
                "action": "create_tile",
                "task_name": "dungeon_floor",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "dark dungeon stone floor tile, moss cracks, pixel art RPG",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 dungeon tile — dark stone, moss, 10–16 colours",
            },
            {
                "action": "create_tile",
                "task_name": "sand_tile",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "sandy desert ground tile, warm yellow-orange, subtle grain texture, pixel art",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 sand tile — warm tones, grain texture",
            },
            {
                "action": "create_tile",
                "task_name": "snow_tile",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "white snowy ground tile, pixel art RPG, soft blue shading",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 snow tile — whites and light blues",
            },
        ],
    },

    # ── Level 3: Characters ────────────────────────────────────────────────
    {
        "level": 3,
        "name": "Characters",
        "min_score": 55,
        "description": "Generate 32×32 character sprites. Nail silhouette and palette.",
        "tasks": [
            {
                "action": "create_character",
                "task_name": "warrior",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art RPG warrior character, sword and shield, top-down view, facing south, 32x32",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 warrior — clear silhouette, sword+shield visible",
            },
            {
                "action": "create_character",
                "task_name": "mage",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art RPG mage wizard, purple robe, magic staff, top-down view, 32x32",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 mage — purple robe, staff, distinct silhouette",
            },
            {
                "action": "create_character",
                "task_name": "villager",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art RPG friendly villager NPC, peasant clothes, brown hair, top-down view, 32x32",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 villager — peasant look, warm colours",
            },
            {
                "action": "create_character",
                "task_name": "slime_enemy",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art cute slime monster RPG enemy, blue jelly body, big eyes, top-down view, 32x32",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 slime — blue jelly, readable at small size",
            },
        ],
    },

    # ── Level 4: Detail & Props ────────────────────────────────────────────
    {
        "level": 4,
        "name": "Detail & Props",
        "min_score": 65,
        "description": "Game props, environment objects, and bigger detailed characters.",
        "tasks": [
            {
                "action": "create_tile",
                "task_name": "treasure_chest",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art wooden treasure chest, gold trim, RPG top-down view, detailed shading, 32x32",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 chest — wood+gold, clear shape, detailed shading",
            },
            {
                "action": "create_tile",
                "task_name": "oak_tree",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art oak tree, RPG top-down map view, lush round canopy, brown trunk, 48x48",
                    "image_size": {"width": 48, "height": 48},
                },
                "target_description": "48×48 oak tree — round canopy, trunk visible, 10+ colours",
            },
            {
                "action": "create_character",
                "task_name": "archer",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art RPG archer ranger, bow and quiver, green cloak, top-down facing south, 32x32",
                    "image_size": {"width": 32, "height": 32},
                },
                "target_description": "32×32 archer — bow+quiver visible, green tone",
            },
            {
                "action": "create_character",
                "task_name": "dragon_boss",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": "pixel art dragon boss enemy, red scales, wings spread, fierce, top-down view, 48x48",
                    "image_size": {"width": 48, "height": 48},
                },
                "target_description": "48×48 dragon — scales, wings, red palette, imposing",
            },
        ],
    },

    # ── Level 5: Expert ────────────────────────────────────────────────────
    {
        "level": 5,
        "name": "Expert — AAA Assets",
        "min_score": 78,
        "description": "Polished game-ready assets matching top indie pixel art standards.",
        "tasks": [
            {
                "action": "create_character",
                "task_name": "hero_protagonist",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": (
                        "pixel art isekai RPG hero, young adventurer, detailed armour, "
                        "glowing blue magic sword, top-down view facing south, AAA quality, 48x48"
                    ),
                    "image_size": {"width": 48, "height": 48},
                },
                "target_description": "48×48 hero — detailed armour, glowing sword, AAA readability",
            },
            {
                "action": "create_tile",
                "task_name": "village_house",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": (
                        "pixel art cosy medieval village house, detailed roof tiles, "
                        "chimney smoke, warm window glow, top-down RPG view, 64x64"
                    ),
                    "image_size": {"width": 64, "height": 64},
                },
                "target_description": "64×64 house — roof, chimney, window glow, high detail",
            },
            {
                "action": "create_character",
                "task_name": "demon_lord",
                "endpoint": "/generate-image-v2",
                "payload": {
                    "description": (
                        "pixel art isekai demon lord final boss, dark armour with glowing red gems, "
                        "menacing aura, top-down view facing south, highly detailed, 64x64"
                    ),
                    "image_size": {"width": 64, "height": 64},
                },
                "target_description": "64×64 boss — dark armour, glowing accents, imposing silhouette",
            },
        ],
    },
]


# ── Image Evaluator ────────────────────────────────────────────────────────────

def evaluate_pixellab_image(b64_image: str, target_description: str = "") -> dict:
    """
    Score a PixelLab-generated image (0–100) on:
      - colour_count  : variety of colours (2–32 ideal)
      - contrast      : dark vs light range
      - edge_clarity  : clean edges (unique border colours)
      - overall       : weighted total
    """
    if not HAS_PIL or not b64_image:
        return {"overall": 50.0, "note": "PIL unavailable — using default score"}

    try:
        raw = base64.b64decode(b64_image)
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        return {"overall": 40.0, "note": f"image decode error: {e}"}

    pixels = list(img.getdata())
    rgb_pixels = [(r, g, b) for r, g, b, a in pixels if a > 10]

    if len(rgb_pixels) < 4:
        return {"overall": 10.0, "note": "mostly transparent"}

    # ── Colour count score ─────────────────────────────────────────────────
    unique_colours = len(set(rgb_pixels))
    if unique_colours < 2:
        colour_score = 5.0
    elif unique_colours <= 6:
        colour_score = 70.0 + (unique_colours - 2) * 5
    elif unique_colours <= 20:
        colour_score = 90.0 + min(unique_colours - 6, 14) * 0.5
    elif unique_colours <= 64:
        colour_score = 95.0 - (unique_colours - 20) * 0.3
    else:
        colour_score = max(50.0, 95.0 - (unique_colours - 64) * 0.5)

    # ── Contrast score ─────────────────────────────────────────────────────
    luminances = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in rgb_pixels]
    lum_range = max(luminances) - min(luminances) if luminances else 0
    contrast_score = min(100.0, (lum_range / 255.0) * 120)

    # ── Edge clarity ───────────────────────────────────────────────────────
    w, h = img.size
    edge_pixels = []
    for x in range(w):
        edge_pixels.append(img.getpixel((x, 0))[:3])
        edge_pixels.append(img.getpixel((x, h - 1))[:3])
    for y in range(h):
        edge_pixels.append(img.getpixel((0, y))[:3])
        edge_pixels.append(img.getpixel((w - 1, y))[:3])
    edge_unique = len(set(edge_pixels))
    # Good pixel art has clean edges: few edge colours, no anti-aliasing noise
    edge_score = max(0.0, 100.0 - max(0, edge_unique - 8) * 3.0)

    # ── Final weighted score ───────────────────────────────────────────────
    overall = colour_score * 0.40 + contrast_score * 0.35 + edge_score * 0.25

    return {
        "overall": round(overall, 1),
        "colour_count": unique_colours,
        "colour_score": round(colour_score, 1),
        "contrast_score": round(contrast_score, 1),
        "edge_score": round(edge_score, 1),
    }


# ── Episode record ─────────────────────────────────────────────────────────────

@dataclass
class PixelEpisode:
    episode: int
    level: int
    level_name: str
    task_name: str
    action: str
    prompt_used: str
    score: float
    breakdown: dict
    image_path: str = ""
    penalty: bool = False
    reward: bool = False


# ── Prompt Improver ────────────────────────────────────────────────────────────

# Vocabulary AI Dev unlocks as it levels up
PROMPT_UPGRADES = {
    1: [],
    2: ["detailed", "clean edges", "vibrant colours"],
    3: ["distinct silhouette", "readable at small size", "strong contrast"],
    4: ["high detail", "AAA pixel art quality", "professional game asset"],
    5: ["masterpiece pixel art", "top-tier indie game standard", "Octopath Traveler quality"],
}

def improve_prompt(base_prompt: str, level: int, history: list[PixelEpisode]) -> str:
    """
    Upgrade the prompt based on current level and past failures.
    Level 1: use base prompt as-is.
    Level 2+: add quality keywords and fix known weaknesses.
    """
    upgrades = []
    for lvl in range(2, level + 1):
        upgrades.extend(PROMPT_UPGRADES.get(lvl, []))

    # Learn from penalties: if last same-type attempt scored low, add fixes
    last_fail = next(
        (ep for ep in reversed(history) if ep.action == "create_tile" and ep.score < 50),
        None
    )
    if last_fail and level >= 2:
        if last_fail.breakdown.get("colour_score", 100) < 60:
            upgrades.append("limited clean palette")
        if last_fail.breakdown.get("contrast_score", 100) < 50:
            upgrades.append("strong light and dark contrast")
        if last_fail.breakdown.get("edge_score", 100) < 60:
            upgrades.append("sharp clean edges no anti-aliasing")

    if upgrades:
        extras = ", ".join(upgrades[:4])  # cap at 4 additions
        return f"{base_prompt}, {extras}"
    return base_prompt


# ── Main Trainer ───────────────────────────────────────────────────────────────

class PixelLabTrainer:
    """
    AI Dev curriculum trainer using PixelLab API for pixel art generation.

    Usage:
        config = PixelLabConfig(api_key="your-key")
        trainer = PixelLabTrainer(config)
        stats = asyncio.run(trainer.run())
    """

    def __init__(
        self,
        config: PixelLabConfig,
        on_episode: Optional[Callable] = None,
    ):
        self.config = config
        self.on_episode = on_episode
        self._history: list[PixelEpisode] = []
        self._running = False
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._memory_path = Path("training_data/pixellab_memory.json")
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_memory()

    # ── State ──────────────────────────────────────────────────────────────

    def _load_memory(self):
        if self._memory_path.exists():
            try:
                data = json.loads(self._memory_path.read_text())
                self._history = [PixelEpisode(**ep) for ep in data]
                print(f"[PixelLab Trainer] Loaded {len(self._history)} past episodes")
            except Exception:
                self._history = []

    def _save_memory(self):
        self._memory_path.write_text(
            json.dumps([asdict(ep) for ep in self._history[-500:]], indent=2)
        )

    @property
    def rolling_avg(self) -> float:
        recent = self._history[-self.config.window_size:]
        return sum(ep.score for ep in recent) / len(recent) if recent else 0.0

    @property
    def current_level(self) -> dict:
        avg = self.rolling_avg
        for lvl in reversed(PIXELLAB_CURRICULUM):
            if avg >= lvl["min_score"]:
                return lvl
        return PIXELLAB_CURRICULUM[0]

    def stop(self):
        self._running = False

    # ── API call ───────────────────────────────────────────────────────────

    def _call_pixellab(self, endpoint: str, payload: dict) -> Optional[str]:
        """
        POST to PixelLab v2 API, poll until complete, return base64 image string.
        v2 returns 202 with background_job_id — must poll /v2/background-jobs/{id}.
        """
        url = PIXELLAB_BASE_V2 + endpoint
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # v2 async: poll job until done
            job_id = data.get("background_job_id")
            if job_id:
                return self._poll_job(job_id, headers)

            # v1-style sync response (fallback)
            if "image" in data:
                img = data["image"]
                return img.get("base64") or img.get("url") or img
            if "images" in data and data["images"]:
                img = data["images"][0]
                return (img.get("base64") or img.get("url") or img) if isinstance(img, dict) else img
            return None

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            msg = e.response.text[:200]
            if code == 402:
                print(f"   ⚠️  PixelLab: Insufficient credits (402)")
            else:
                print(f"   ⚠️  PixelLab API error {code}: {msg}")
            return None
        except Exception as e:
            print(f"   ⚠️  PixelLab request failed: {e}")
            return None

    def _poll_job(self, job_id: str, headers: dict, timeout: int = 300) -> Optional[str]:
        """Poll /v2/background-jobs/{job_id} until completed, return base64 image."""
        poll_url = f"{PIXELLAB_BASE_V2}/background-jobs/{job_id}"
        deadline = time.time() + timeout
        interval = 3
        while time.time() < deadline:
            time.sleep(interval)
            interval = min(interval + 2, 10)  # back-off up to 10s
            try:
                r = requests.get(poll_url, headers=headers, timeout=30)
                r.raise_for_status()
                d = r.json()
                status = d.get("status")
                if status == "completed":
                    # Images live in last_response.images
                    lr = d.get("last_response", {})
                    imgs = lr.get("images", [])
                    if imgs:
                        img = imgs[0]
                        if isinstance(img, dict):
                            return img.get("base64") or img.get("url")
                        return img
                    return None
                elif status in ("failed", "cancelled"):
                    print(f"   ⚠️  PixelLab job {status}: {d.get('error','')}")
                    return None
                # still processing
                progress = d.get("last_response", {}).get("progress", 0)
                print(f"   ⏳ {int(progress*100)}%", end="\r", flush=True)
            except Exception as e:
                print(f"   ⚠️  Poll error: {e}")
                break
        print(f"   ⚠️  Job {job_id} timed out after {timeout}s")
        return None

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> dict:
        self._running = True
        episode_count = len(self._history)
        total_new = 0

        print(f"\n{'='*60}")
        print("🎨 AI Dev — PixelLab Pixel Art Training")
        print(f"   API endpoint : {PIXELLAB_BASE_V2}")
        print(f"   Max episodes : {self.config.max_episodes}")
        print(f"   Target score : {self.config.target_score}")
        print(f"   Output dir   : {self._output_dir}")
        print(f"{'='*60}\n")
        print("Curriculum progression:")
        for lvl in PIXELLAB_CURRICULUM:
            print(f"  Level {lvl['level']} ({lvl['min_score']}+): {lvl['name']} — {lvl['description']}")
        print()

        for _ in range(self.config.max_episodes):
            if not self._running:
                print("[Trainer] Stopped.")
                break

            episode_count += 1
            total_new += 1
            curr = self.current_level
            task = random.choice(curr["tasks"])

            print(f"\n── Episode {episode_count} [Level {curr['level']}: {curr['name']}] ──")
            print(f"   Task   : {task['task_name']}")
            print(f"   Action : {task['action']}")

            # Build improved prompt based on history + level
            base_payload = dict(task["payload"])
            improved_desc = improve_prompt(
                base_payload.get("description", ""),
                curr["level"],
                self._history,
            )
            base_payload["description"] = improved_desc
            print(f"   Prompt : {improved_desc[:80]}...")

            # Call PixelLab API
            t0 = time.time()
            b64 = self._call_pixellab(task["endpoint"], base_payload) or ""
            elapsed = time.time() - t0
            print(f"   API    : {elapsed:.1f}s")

            # Evaluate
            breakdown = evaluate_pixellab_image(b64, task.get("target_description", ""))
            score = breakdown["overall"]

            # Reward / penalty feedback
            is_reward  = score >= self.config.reward_threshold
            is_penalty = score < self.config.penalty_threshold
            if is_reward:
                print(f"   ✅ REWARD  — Score: {score}/100  (rolling avg: {self.rolling_avg:.1f})")
            elif is_penalty:
                print(f"   ❌ PENALTY — Score: {score}/100  (rolling avg: {self.rolling_avg:.1f})")
            else:
                print(f"   ⚠️  OK     — Score: {score}/100  (rolling avg: {self.rolling_avg:.1f})")

            # Save image
            img_path = ""
            if self.config.save_images and b64:
                img_path = str(
                    self._output_dir / f"ep{episode_count:04d}_L{curr['level']}_{task['task_name']}.png"
                )
                try:
                    raw_bytes = base64.b64decode(b64)
                    Path(img_path).write_bytes(raw_bytes)
                    print(f"   💾 Saved : {img_path}")
                except Exception:
                    img_path = ""

            # Store episode
            ep = PixelEpisode(
                episode=episode_count,
                level=curr["level"],
                level_name=curr["name"],
                task_name=task["task_name"],
                action=task["action"],
                prompt_used=improved_desc,
                score=score,
                breakdown=breakdown,
                image_path=img_path,
                penalty=is_penalty,
                reward=is_reward,
            )
            self._history.append(ep)
            self._save_memory()

            # Callback for UI / Colab
            if self.on_episode:
                self.on_episode({
                    "episode": episode_count,
                    "level": curr["level"],
                    "level_name": curr["name"],
                    "task_name": task["task_name"],
                    "score": score,
                    "breakdown": breakdown,
                    "rolling_avg": self.rolling_avg,
                    "image_path": img_path,
                    "image_b64": b64,
                    "prompt": improved_desc,
                    "reward": is_reward,
                    "penalty": is_penalty,
                })

            # Check target
            if (
                total_new >= self.config.window_size
                and self.rolling_avg >= self.config.target_score
            ):
                print(
                    f"\n🏆 TARGET REACHED — rolling avg {self.rolling_avg:.1f} ≥ {self.config.target_score}"
                )
                break

            # Small delay to respect API rate limits
            await asyncio.sleep(1)

        scores = [ep.score for ep in self._history[-total_new:]] if total_new else []
        return {
            "total_episodes": episode_count,
            "new_this_run": total_new,
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
            "best": round(max(scores), 1) if scores else 0.0,
            "worst": round(min(scores), 1) if scores else 0.0,
            "final_level": self.current_level["level"],
            "final_level_name": self.current_level["name"],
            "output_dir": str(self._output_dir),
        }

    # ── Summary helpers ────────────────────────────────────────────────────

    def print_progress_report(self):
        if not self._history:
            print("No training history yet.")
            return
        print(f"\n{'='*60}")
        print("📊 PixelLab Training Progress Report")
        print(f"   Total episodes : {len(self._history)}")
        print(f"   Current level  : {self.current_level['level']} — {self.current_level['name']}")
        print(f"   Rolling avg    : {self.rolling_avg:.1f}/100")
        recent = self._history[-10:]
        print(f"   Last 10 scores : {[ep.score for ep in recent]}")
        best = max(self._history, key=lambda e: e.score)
        print(f"   Best ever      : {best.score}/100 — {best.task_name} (ep {best.episode})")
        print(f"{'='*60}\n")


# ── Convenience entry point ────────────────────────────────────────────────────

async def run_pixellab_training(
    api_key: str = "",
    episodes: int = 50,
    target: float = 80.0,
    on_episode: Optional[Callable] = None,
) -> dict:
    """Simple entry point for Colab notebooks and CLI."""
    cfg = PixelLabConfig(api_key=api_key, max_episodes=episodes, target_score=target)
    trainer = PixelLabTrainer(cfg, on_episode=on_episode)
    return await trainer.run()
