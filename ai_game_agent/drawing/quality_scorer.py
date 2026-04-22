"""
Quality Scorer — evaluates AI Dev's pixel art against GOTY standards.

Scores on multiple criteria:
  - Color discipline   (palette count, harmony, saturation)
  - Depth / 2.75D     (value contrast across image, layer separation)
  - Detail density     (not too sparse, not noisy)
  - Edge clarity       (clean pixel edges, no anti-aliasing blur)
  - Value contrast     (light vs dark ratio — readability)
  - Composition        (color mass distribution, focal point)
  - Coverage           (how much of canvas is filled vs empty/default)
  - Uniqueness         (how different from a blank canvas)

Returns 0-100 overall and per-criteria breakdown.
"""
from __future__ import annotations
import colorsys
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def score_drawing(image: Image.Image, task: Dict = None) -> Dict[str, float]:
    """
    Score a completed pixel art drawing.
    Returns dict with 'overall' (0-100) and individual criteria scores.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    scores = {}
    scores["coverage"]       = _score_coverage(arr)
    scores["color_discipline"]= _score_color_discipline(arr, img)
    scores["value_contrast"] = _score_value_contrast(arr)
    scores["depth_illusion"] = _score_depth_illusion(arr)
    scores["edge_clarity"]   = _score_edge_clarity(arr)
    scores["detail_density"] = _score_detail_density(arr)
    scores["composition"]    = _score_composition(arr)
    scores["palette_harmony"]= _score_palette_harmony(img)

    # Weighted overall — GOTY demands everything
    weights = {
        "coverage":        0.10,
        "color_discipline":0.15,
        "value_contrast":  0.15,
        "depth_illusion":  0.20,  # 2.75D is key
        "edge_clarity":    0.10,
        "detail_density":  0.10,
        "composition":     0.10,
        "palette_harmony": 0.10,
    }
    overall = sum(scores[k] * weights[k] for k in weights) / sum(weights.values())
    scores["overall"] = round(overall, 1)
    return scores


# ── Individual criteria ────────────────────────────────────────────────────

def _score_coverage(arr: np.ndarray) -> float:
    """How much of the canvas has color (not default background)."""
    h, w, _ = arr.shape
    bg = arr[0, 0]  # top-left pixel as background reference
    diff = np.abs(arr - bg).sum(axis=2)
    filled = (diff > 15).sum() / (h * w)
    # 30%+ filled = good; 80%+ = great
    if filled < 0.10:
        return 10.0
    if filled < 0.30:
        return 30.0 + filled * 100
    if filled < 0.60:
        return 60.0 + (filled - 0.30) * 100
    return min(100.0, 70.0 + filled * 40)


def _score_color_discipline(arr: np.ndarray, img: Image.Image) -> float:
    """
    Score palette discipline: unique color count, consistency.
    GOTY: 8-24 unique colors. Too few = boring. Too many = chaotic.
    """
    # Quantize to count effective unique colors
    q = img.quantize(colors=256)
    hist = q.histogram()
    used = sum(1 for v in hist if v > 2)  # colors used more than 2px

    if used == 0:
        return 0.0
    if used <= 3:
        return 20.0
    if used <= 8:
        return 60.0 + (used - 4) * 5
    if used <= 24:
        return 80.0 + min(20.0, (24 - used) * 1.5)
    if used <= 48:
        return max(50.0, 80.0 - (used - 24) * 1.5)
    return max(20.0, 50.0 - (used - 48) * 0.5)


def _score_value_contrast(arr: np.ndarray) -> float:
    """
    Score light-vs-dark contrast. Strong contrast = readable, GOTY quality.
    Use luminance distribution.
    """
    lum = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    lum_norm = lum / 255.0

    dark_ratio  = (lum_norm < 0.25).mean()
    light_ratio = (lum_norm > 0.75).mean()
    mid_ratio   = ((lum_norm >= 0.25) & (lum_norm <= 0.75)).mean()
    std         = lum_norm.std()

    # Good contrast: dark + light both present, high std
    has_dark  = dark_ratio  > 0.05
    has_light = light_ratio > 0.05
    has_mid   = mid_ratio   > 0.15

    score = 0.0
    if has_dark:  score += 25
    if has_light: score += 25
    if has_mid:   score += 20
    score += min(30.0, std * 150)  # std typically 0.1-0.35

    return min(100.0, score)


def _score_depth_illusion(arr: np.ndarray) -> float:
    """
    Score 2.75D depth: top of image should be lighter/less saturated than bottom.
    This simulates atmospheric perspective (far=light, near=dark).
    """
    h, w, _ = arr.shape
    if h < 8:
        return 50.0

    # Divide into horizontal thirds: top (bg), middle (mid), bottom (fg)
    top    = arr[:h//3, :, :]
    middle = arr[h//3:2*h//3, :, :]
    bottom = arr[2*h//3:, :, :]

    def lum_mean(region):
        return (0.299*region[:,:,0] + 0.587*region[:,:,1] + 0.114*region[:,:,2]).mean()

    def sat_mean(region):
        # Rough saturation: range of RGB channels
        rng = region.max(axis=2) - region.min(axis=2)
        return rng.mean()

    top_lum, mid_lum, bot_lum = lum_mean(top), lum_mean(middle), lum_mean(bottom)
    top_sat, bot_sat = sat_mean(top), sat_mean(bottom)

    score = 0.0

    # Atmospheric: top brighter than bottom (lighter sky/bg)
    if top_lum > bot_lum:
        diff = (top_lum - bot_lum) / 255.0
        score += min(35.0, diff * 200)

    # Top less saturated (far = desaturated)
    if top_sat < bot_sat:
        diff = (bot_sat - top_sat) / 255.0
        score += min(25.0, diff * 150)

    # Value variety across thirds (all three different)
    vals = sorted([top_lum, mid_lum, bot_lum])
    spread = vals[2] - vals[0]
    score += min(25.0, spread / 255.0 * 200)

    # Bonus: bottom darkest (ground plane)
    if bot_lum < top_lum and bot_lum < mid_lum:
        score += 15.0

    return min(100.0, score)


def _score_edge_clarity(arr: np.ndarray) -> float:
    """
    Score pixel art edge quality. Real pixel art has hard, aliased edges.
    Blurry/anti-aliased art scores low. Sharp transitions score high.
    """
    # Measure horizontal and vertical edge sharpness
    h, w, _ = arr.shape
    if h < 4 or w < 4:
        return 50.0

    lum = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2])

    # Horizontal differences
    hdiff = np.abs(np.diff(lum, axis=1))
    # Vertical differences
    vdiff = np.abs(np.diff(lum, axis=0))

    # Sharp pixel art: most differences are either 0 (same) or large (edge)
    # Anti-aliased: lots of small differences (2-30 range)
    all_diff = np.concatenate([hdiff.flatten(), vdiff.flatten()])

    zero_pct  = (all_diff < 3).mean()    # same color neighbors
    sharp_pct = (all_diff > 30).mean()   # actual hard edges
    fuzzy_pct = ((all_diff >= 3) & (all_diff <= 30)).mean()  # anti-alias blur

    score = 0.0
    score += min(40.0, zero_pct * 50)    # lots of flat regions = good
    score += min(40.0, sharp_pct * 200)  # hard edges = pixel art quality
    score -= min(40.0, fuzzy_pct * 80)   # penalize anti-aliasing
    score = max(0.0, score)

    return min(100.0, score)


def _score_detail_density(arr: np.ndarray) -> float:
    """
    Score detail: not too sparse (boring) and not too noisy (chaotic).
    Target: 15-60% of pixels differ from their neighbor.
    """
    lum = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2])
    hdiff = np.abs(np.diff(lum, axis=1)) > 10
    change_rate = hdiff.mean()

    # Ideal range: 10-50% change rate
    if change_rate < 0.05:
        return 20.0 + change_rate * 200   # too sparse
    if change_rate < 0.10:
        return 50.0 + (change_rate - 0.05) * 400
    if change_rate < 0.50:
        return min(100.0, 70.0 + (change_rate - 0.10) * 75)
    if change_rate < 0.70:
        return max(50.0, 100.0 - (change_rate - 0.50) * 150)  # getting noisy
    return max(20.0, 50.0 - (change_rate - 0.70) * 100)  # too noisy


def _score_composition(arr: np.ndarray) -> float:
    """
    Score composition: focal point exists, not all one color, color mass balanced.
    """
    h, w, _ = arr.shape
    lum = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2])

    # Divide into 3×3 grid and measure variance across sections
    sec_h, sec_w = max(1, h//3), max(1, w//3)
    section_means = []
    for r in range(3):
        for c in range(3):
            sec = lum[r*sec_h:(r+1)*sec_h, c*sec_w:(c+1)*sec_w]
            section_means.append(sec.mean())

    variance = np.var(section_means)
    global_std = lum.std()

    score = 0.0
    # High variance across sections = interesting composition
    score += min(50.0, variance / 100)
    # High global std = good value range
    score += min(50.0, global_std / 255 * 200)

    return min(100.0, score)


def _score_palette_harmony(img: Image.Image) -> float:
    """
    Score color harmony: do the colors feel intentional and related?
    Uses hue clustering — harmonious palettes cluster around 2-4 hue groups.
    """
    # Get all pixel colors
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    pixels = arr.reshape(-1, 3)

    # Sample for speed
    if len(pixels) > 5000:
        idx = np.random.choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]

    # Convert to HSV
    hues = []
    for px in pixels:
        r, g, b = px[0]/255, px[1]/255, px[2]/255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s > 0.15 and v > 0.10:  # Only count non-grey, non-black pixels
            hues.append(h)

    if len(hues) < 50:
        return 50.0  # Not enough color info

    hues = np.array(hues)

    # Count hue clusters (bins of 30 degrees = 12 bins)
    hist, _ = np.histogram(hues, bins=12, range=(0, 1))
    occupied_bins = (hist > len(hues) * 0.03).sum()  # bins with >3% of hues

    # Ideal: 2-5 distinct hue groups (analogous or complementary palette)
    if occupied_bins == 0:
        return 20.0
    if occupied_bins <= 2:
        return 70.0  # Monochromatic/analogous — clean
    if occupied_bins <= 4:
        return 90.0  # Complementary/triadic — very harmonious
    if occupied_bins <= 6:
        return 70.0  # Split-complementary — still good
    if occupied_bins <= 8:
        return 50.0  # Getting muddy
    return max(20.0, 50.0 - (occupied_bins - 8) * 5)  # Too many hues = chaotic


def generate_feedback(scores: Dict[str, float], task: Dict = None) -> str:
    """Generate human-readable feedback for AI Dev from scores."""
    lines = []
    overall = scores.get("overall", 0)

    if overall >= 90:
        lines.append("🏆 MASTERPIECE — GOTY level quality achieved!")
    elif overall >= 78:
        lines.append("✅ Excellent work — professional quality, near-perfect.")
    elif overall >= 65:
        lines.append("⚠️ Good progress — but not yet at GOTY standard.")
    elif overall >= 50:
        lines.append("❌ Mediocre — significant improvements needed.")
    else:
        lines.append("💀 PENALTY — Poor quality. Major rework required.")

    # Identify weakest areas
    criteria = {k: v for k, v in scores.items() if k != "overall"}
    sorted_criteria = sorted(criteria.items(), key=lambda x: x[1])

    lines.append(f"Weakest: {sorted_criteria[0][0]} ({sorted_criteria[0][1]:.0f}/100) — fix this first.")
    if sorted_criteria[1][1] < 70:
        lines.append(f"Also weak: {sorted_criteria[1][0]} ({sorted_criteria[1][1]:.0f}/100).")

    # Specific tips for worst scores
    worst_name, worst_score = sorted_criteria[0]
    tips = {
        "coverage":        "Draw more! Fill the canvas — empty space is wasted potential.",
        "color_discipline":"Reduce palette to 8-16 colors. Every color must serve a purpose.",
        "value_contrast":  "Add strong darks AND lights. The image needs contrast to pop.",
        "depth_illusion":  "Use 3 layers! Background light+desaturated, foreground dark+saturated.",
        "edge_clarity":    "Hard pixel edges only! No gradients on solid shapes.",
        "detail_density":  "Balance detail — some flat areas for rest, some busy for focal points.",
        "composition":     "Create a focal point! One area should draw the eye.",
        "palette_harmony": "Pick 2-3 related hues. All colors should feel like a family.",
    }
    if worst_name in tips:
        lines.append(f"Fix: {tips[worst_name]}")

    return " ".join(lines)
