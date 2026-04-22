"""
Game Quality Evaluator
======================
Scores generated games on three axes:
  - Pixel Art Quality  (0–100)
  - Code Quality       (0–100)
  - Game Design        (0–100)

Applies automatic penalties for bad outputs so the RL trainer
knows exactly what to penalise.

Final score = weighted average of three axes.
"""

from __future__ import annotations

import base64
import io
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Palette definitions (for adherence check) ─────────────────────────────────

DB32_PALETTE = [
    (0,0,0),(34,32,52),(69,40,60),(102,57,49),(143,86,59),(223,113,38),
    (217,160,102),(238,195,154),(251,242,54),(153,229,80),(106,190,48),
    (55,148,110),(75,105,47),(82,75,36),(50,60,57),(63,63,116),(48,96,130),
    (91,110,225),(99,155,255),(95,205,228),(203,219,252),(255,255,255),
    (155,173,183),(132,126,135),(105,106,106),(89,86,82),(118,66,138),
    (172,50,50),(217,87,99),(215,123,186),(143,151,74),(138,111,48)
]
PICO8_PALETTE = [
    (0,0,0),(29,43,83),(126,37,83),(0,135,81),(171,82,54),(95,87,79),
    (194,195,199),(255,241,232),(255,0,77),(255,163,0),(255,236,39),
    (0,228,54),(41,173,255),(131,118,156),(255,119,168),(255,204,170)
]

def _nearest_palette_dist(rgb: tuple, palette: list) -> float:
    r, g, b = rgb
    best = float('inf')
    for pr, pg, pb in palette:
        d = (r-pr)**2 + (g-pg)**2 + (b-pb)**2
        if d < best:
            best = d
    return math.sqrt(best)


# ── Score result dataclass ────────────────────────────────────────────────────

@dataclass
class EvalResult:
    pixel_art_score: float = 0.0
    code_score: float = 0.0
    design_score: float = 0.0
    total_score: float = 0.0
    penalties: list[str] = field(default_factory=list)
    bonuses: list[str] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Total: {self.total_score:.1f}/100",
            f"  Pixel Art : {self.pixel_art_score:.1f}/100",
            f"  Code      : {self.code_score:.1f}/100",
            f"  Design    : {self.design_score:.1f}/100",
        ]
        if self.bonuses:
            lines.append("  ✅ " + " | ".join(self.bonuses))
        if self.penalties:
            lines.append("  ❌ " + " | ".join(self.penalties))
        return "\n".join(lines)


# ── Pixel Art Evaluator ───────────────────────────────────────────────────────

class PixelArtEvaluator:
    """
    Scores a single PIL Image on pixel art quality.
    Rewards: palette adherence, contrast, clean edges, appropriate colour count.
    Penalises: fully black/white, noise, too many colours, zero contrast.
    """

    def evaluate(self, image: "Image.Image") -> tuple[float, list[str], list[str]]:
        if not HAS_PIL:
            return 50.0, [], ["PIL not available — using default score"]

        penalties, bonuses = [], []
        score = 0.0

        img = image.convert("RGBA")
        w, h = img.size
        pixels = list(img.getdata())
        rgb_pixels = [(r, g, b) for r, g, b, a in pixels if a > 10]

        if not rgb_pixels:
            return 0.0, ["Image is fully transparent"], []

        # ── 1. Colour count (20 pts) ─────────────────────────────────────────
        unique_colors = set(rgb_pixels)
        n_colors = len(unique_colors)
        if n_colors <= 2:
            score += 0
            penalties.append(f"Only {n_colors} colours (too flat)")
        elif n_colors <= 32:
            score += 20
            bonuses.append(f"Good colour count ({n_colors})")
        elif n_colors <= 64:
            score += 12
        elif n_colors <= 128:
            score += 6
            penalties.append(f"Too many colours ({n_colors}) — not pixel art style")
        else:
            score += 0
            penalties.append(f"Way too many colours ({n_colors}) — not pixel art")

        # ── 2. Palette adherence (20 pts) ────────────────────────────────────
        sample = list(unique_colors)[:50]
        db32_dists = [_nearest_palette_dist(c, DB32_PALETTE) for c in sample]
        pico_dists = [_nearest_palette_dist(c, PICO8_PALETTE) for c in sample]
        avg_db32 = sum(db32_dists) / len(db32_dists) if db32_dists else 999
        avg_pico = sum(pico_dists) / len(pico_dists) if pico_dists else 999
        best_pal_dist = min(avg_db32, avg_pico)
        if best_pal_dist < 20:
            score += 20
            bonuses.append("Excellent palette adherence")
        elif best_pal_dist < 40:
            score += 14
            bonuses.append("Good palette adherence")
        elif best_pal_dist < 70:
            score += 7
        else:
            score += 0
            penalties.append("Colours don't match any known pixel art palette")

        # ── 3. Contrast (20 pts) ─────────────────────────────────────────────
        rs = [r for r, g, b in rgb_pixels]
        gs = [g for r, g, b in rgb_pixels]
        bs = [b for r, g, b in rgb_pixels]
        brightness = [(r*299 + g*587 + b*114) // 1000 for r, g, b in rgb_pixels]
        contrast = max(brightness) - min(brightness)
        if contrast > 180:
            score += 20
            bonuses.append("High contrast — readable")
        elif contrast > 100:
            score += 14
        elif contrast > 50:
            score += 7
        else:
            score += 0
            penalties.append(f"Very low contrast ({contrast}) — unreadable")

        # ── 4. Not blank (20 pts) ────────────────────────────────────────────
        avg_r = sum(rs) / len(rs)
        avg_g = sum(gs) / len(gs)
        avg_b = sum(bs) / len(bs)
        is_all_black = avg_r < 10 and avg_g < 10 and avg_b < 10
        is_all_white = avg_r > 245 and avg_g > 245 and avg_b > 245
        if is_all_black:
            score += 0
            penalties.append("Image is completely black — MAJOR PENALTY")
        elif is_all_white:
            score += 0
            penalties.append("Image is completely white — MAJOR PENALTY")
        else:
            score += 20

        # ── 5. Edge cleanliness (20 pts) ────────────────────────────────────
        # Count isolated single pixels (surrounded by transparency/very different colour)
        if HAS_PIL:
            try:
                import numpy as np
                arr = np.array(img)
                alpha = arr[:, :, 3]
                # Edge = pixels with alpha >10 adjacent to alpha==0
                inner = (alpha[1:-1, 1:-1] > 10)
                top    = (alpha[0:-2, 1:-1] == 0)
                bot    = (alpha[2:,   1:-1] == 0)
                left   = (alpha[1:-1, 0:-2] == 0)
                right  = (alpha[1:-1, 2:]   == 0)
                isolated = inner & top & bot & left & right
                iso_frac = isolated.sum() / max(inner.sum(), 1)
                if iso_frac < 0.02:
                    score += 20
                    bonuses.append("Clean edges")
                elif iso_frac < 0.08:
                    score += 12
                elif iso_frac < 0.20:
                    score += 5
                    penalties.append(f"Noisy edges ({iso_frac:.1%} isolated pixels)")
                else:
                    score += 0
                    penalties.append(f"Very noisy/messy art ({iso_frac:.1%} isolated pixels)")
            except Exception:
                score += 10  # neutral

        return min(score, 100.0), penalties, bonuses


# ── GDScript Code Evaluator ───────────────────────────────────────────────────

class CodeEvaluator:
    """
    Scores GDScript code on correctness and completeness.
    Uses regex heuristics (no Godot binary needed at training time).
    """

    REQUIRED_CALLBACKS = ["_ready", "_process", "_input", "_physics_process", "_unhandled_input"]
    GOOD_PATTERNS = [
        (r"extends\s+\w+", 5, "Proper class extension"),
        (r"class_name\s+\w+", 5, "Named class"),
        (r"signal\s+\w+", 5, "Uses signals"),
        (r"@export", 5, "Uses exports for editor config"),
        (r"CollisionShape|CharacterBody|RigidBody|Area2D", 10, "Has physics nodes"),
        (r"AnimationPlayer|AnimatedSprite", 8, "Has animations"),
        (r"Input\.", 8, "Handles player input"),
        (r"get_tree\(\)|SceneTree", 5, "Uses scene tree"),
        (r"save|load|FileAccess", 5, "Has save/load"),
        (r"score|health|mana|stamina|level", 5, "Has game stats"),
        (r"enum\s+\w+", 5, "Uses enums for state"),
        (r"func _ready", 5, "Has _ready"),
        (r"func _process|func _physics_process", 7, "Has update loop"),
    ]
    BAD_PATTERNS = [
        (r"print\s*\(\s*\)", -3, "Empty print statement"),
        (r"pass\s*\n.*pass\s*\n", -5, "Multiple consecutive pass statements"),
        (r"TODO|FIXME|HACK|XXX", -5, "Unfinished placeholder code"),
        (r"^\s*#.*$\n\s*#.*$\n\s*#.*$\n\s*#.*$", -5, "Wall of comments with no code"),
    ]

    def evaluate(self, code: str) -> tuple[float, list[str], list[str]]:
        penalties, bonuses = [], []
        score = 0.0

        if not code or len(code.strip()) < 10:
            return 0.0, ["Empty or nearly empty code — MAJOR PENALTY"], []

        # Line count sanity
        lines = code.split("\n")
        n_lines = len(lines)
        if n_lines < 5:
            penalties.append(f"Very short code ({n_lines} lines)")
            score -= 20
        elif n_lines >= 30:
            score += 10
            bonuses.append(f"Substantial code ({n_lines} lines)")
        elif n_lines >= 10:
            score += 5

        # Good patterns
        for pattern, pts, label in self.GOOD_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                score += pts
                bonuses.append(label)

        # Bad patterns
        for pattern, pts, label in self.BAD_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                score += pts
                penalties.append(label)

        # Syntax heuristics
        open_brackets = code.count("{") + code.count("(")
        close_brackets = code.count("}") + code.count(")")
        if abs(open_brackets - close_brackets) > 5:
            score -= 15
            penalties.append("Unbalanced brackets — likely syntax error")

        # Indentation check (GDScript is indent-sensitive)
        indented = sum(1 for l in lines if l.startswith("\t") or l.startswith("  "))
        if indented < 2 and n_lines > 5:
            score -= 20
            penalties.append("No indentation — GDScript will fail to parse")

        return max(0.0, min(score, 100.0)), penalties, bonuses


# ── Game Design Evaluator ─────────────────────────────────────────────────────

class DesignEvaluator:
    """
    Scores overall game design from code + metadata.
    Checks for player, enemies, goals, progression, UI.
    """

    DESIGN_CHECKS = [
        (r"player|hero|character|protagonist", 15, "Has player character"),
        (r"enemy|monster|mob|npc|boss", 15, "Has enemies/NPCs"),
        (r"win|victory|complete|finish|goal", 15, "Has win condition"),
        (r"die|death|game.?over|lose|health\s*<=\s*0|hp\s*<=\s*0", 15, "Has lose/death condition"),
        (r"score|xp|experience|level.?up|progress", 10, "Has progression"),
        (r"Label|RichText|HUD|UI|interface", 10, "Has UI elements"),
        (r"map|world|dungeon|room|area|zone|biome", 10, "Has world/map design"),
        (r"item|weapon|armor|potion|loot|drop|inventory", 10, "Has items/inventory"),
    ]
    DESIGN_PENALTIES = [
        (r"^\s*extends Node\s*$", -10, "Extends bare Node — no game structure"),
        (r"^\s*pass\s*$", -5, "Empty script body"),
    ]

    def evaluate(self, code: str, description: str = "") -> tuple[float, list[str], list[str]]:
        penalties, bonuses = [], []
        score = 0.0
        full_text = code + " " + description

        for pattern, pts, label in self.DESIGN_CHECKS:
            if re.search(pattern, full_text, re.IGNORECASE):
                score += pts
                bonuses.append(label)

        for pattern, pts, label in self.DESIGN_PENALTIES:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                score += pts
                penalties.append(label)

        return max(0.0, min(score, 100.0)), penalties, bonuses


# ── Main Evaluator (combines all three) ──────────────────────────────────────

class GameEvaluator:
    """
    Master evaluator — call .evaluate() with any combination of:
      - image (PIL Image or base64 string)
      - code (GDScript string)
      - description (str) for design hints
    Returns EvalResult with total score, breakdown, penalties, bonuses.
    """

    WEIGHTS = {"pixel_art": 0.35, "code": 0.35, "design": 0.30}

    def __init__(self):
        self._pixel = PixelArtEvaluator()
        self._code  = CodeEvaluator()
        self._design = DesignEvaluator()

    def evaluate(
        self,
        image: Optional[object] = None,
        code: str = "",
        description: str = "",
    ) -> EvalResult:
        result = EvalResult()
        all_penalties = []
        all_bonuses = []

        has_image = image is not None
        has_code = bool(code)
        has_desc = bool(description)

        # ── Pixel art score ──────────────────────────────────────────────────
        if has_image:
            pil_img = self._to_pil(image)
            if pil_img:
                pa_score, pa_pen, pa_bon = self._pixel.evaluate(pil_img)
                result.pixel_art_score = pa_score
                all_penalties += [f"[Art] {p}" for p in pa_pen]
                all_bonuses   += [f"[Art] {b}" for b in pa_bon]
                result.breakdown["pixel_art"] = pa_score
            else:
                result.pixel_art_score = 50.0
        else:
            result.pixel_art_score = 50.0

        # ── Code score ──────────────────────────────────────────────────────
        if has_code:
            c_score, c_pen, c_bon = self._code.evaluate(code)
            result.code_score = c_score
            all_penalties += [f"[Code] {p}" for p in c_pen]
            all_bonuses   += [f"[Code] {b}" for b in c_bon]
            result.breakdown["code"] = c_score
        else:
            result.code_score = 50.0

        # ── Design score ─────────────────────────────────────────────────────
        if has_code or has_desc:
            d_score, d_pen, d_bon = self._design.evaluate(code, description)
            result.design_score = d_score
            all_penalties += [f"[Design] {p}" for p in d_pen]
            all_bonuses   += [f"[Design] {b}" for b in d_bon]
            result.breakdown["design"] = d_score
        else:
            result.design_score = 50.0

        # ── Adaptive weights (art-only tasks don't get penalised on design) ─
        if has_image and not has_code:
            # Pure art task: weight art heavily, code/design are neutral fillers
            w = {"pixel_art": 0.70, "code": 0.15, "design": 0.15}
        elif has_code and not has_image:
            # Pure code task: weight code + design, art is neutral
            w = {"pixel_art": 0.10, "code": 0.50, "design": 0.40}
        else:
            # Mixed task
            w = self.WEIGHTS

        result.total_score = (
            result.pixel_art_score * w["pixel_art"]
            + result.code_score    * w["code"]
            + result.design_score  * w["design"]
        )

        result.penalties = all_penalties
        result.bonuses   = all_bonuses
        return result

    def _to_pil(self, image) -> Optional["Image.Image"]:
        if not HAS_PIL:
            return None
        if isinstance(image, str):
            # base64 PNG
            try:
                data = base64.b64decode(image)
                return Image.open(io.BytesIO(data))
            except Exception:
                return None
        try:
            return image  # already PIL Image
        except Exception:
            return None
