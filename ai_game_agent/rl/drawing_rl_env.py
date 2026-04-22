"""
Drawing RL Environment — AI Dev discovers great pixel art through pure trial/error.

Instead of asking Gemini "what should I draw?", the PPO agent:
  - Observes: current canvas state (color histogram + coverage features)
  - Actions: pick from a palette of pixel-drawing operations
  - Reward: improvement in art quality score

This is how DeepMind's agents learned to play games —
pure self-discovery, no human telling it what to draw.

State vector (64 floats):
  - 48: RGB histogram (16 bins × 3 channels)
  - 8:  spatial coverage quadrants
  - 4:  mean/std of brightness/saturation
  - 4:  edge density (how detailed the drawing is)

Actions (32 discrete):
  - 8 palette color choices × 4 brush sizes = 32 combinations
  - Agent picks WHERE to draw via a learned position preference
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List

try:
    from PIL import Image, ImageDraw
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── Drawing primitives ────────────────────────────────────────────────────────

# 8 palette slots — AI Dev learns which colors to use for quality pixel art
PALETTE_COLORS = [
    (20, 12, 28),    # deep shadow
    (68, 36, 52),    # dark mid
    (48, 96, 130),   # blue mid
    (91, 110, 225),  # highlight blue
    (99, 155, 255),  # sky
    (251, 242, 54),  # accent yellow
    (255, 163, 0),   # warm highlight
    (255, 255, 255), # pure white
]

BRUSH_SIZES = [1, 2, 3, 4]          # pixels
N_ACTIONS   = len(PALETTE_COLORS) * len(BRUSH_SIZES)   # 32

# Grid positions the agent can draw at (8×8 grid on a 32px canvas)
GRID_SIZE   = 4
CANVAS_SIZE = 32


class DrawingEnv:
    """
    Gym-style environment for drawing RL.

    Each episode = one sprite drawing attempt.
    Max steps per episode = 64 (enough to cover a 32×32 canvas).
    """

    def __init__(self, canvas_size: int = CANVAS_SIZE, max_steps: int = 64):
        self.canvas_size = canvas_size
        self.max_steps   = max_steps
        self.obs_dim     = 64
        self.action_dim  = N_ACTIONS

        self._canvas: Optional[np.ndarray] = None   # RGBA uint8
        self._step   = 0
        self._prev_score = 0.0
        self._draw_pos_idx = 0   # cycles through grid positions

        # Scoring function (injected from animation_scorer)
        self._score_fn = None

    def set_scorer(self, fn):
        """Inject a scoring function: fn(canvas_rgba_array) -> float 0-100."""
        self._score_fn = fn

    def reset(self) -> np.ndarray:
        if PIL_OK:
            img = Image.new("RGBA", (self.canvas_size, self.canvas_size), (0, 0, 0, 0))
            self._canvas = np.array(img, dtype=np.uint8)
        else:
            self._canvas = np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.uint8)
        self._step = 0
        self._prev_score = 0.0
        self._draw_pos_idx = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self._canvas is not None, "Call reset() first"

        color_idx = action // len(BRUSH_SIZES)
        brush_idx = action % len(BRUSH_SIZES)
        color     = PALETTE_COLORS[color_idx]
        brush     = BRUSH_SIZES[brush_idx]

        # Spiral drawing position (agent cycles through canvas)
        x, y = self._next_position()
        self._draw_pixel(x, y, color, brush)

        self._step += 1
        done  = self._step >= self.max_steps
        score = self._compute_score()
        reward = (score - self._prev_score) / 10.0   # normalize to ~[-1, 1]

        # Small penalty for drawing outside interesting regions
        if self._is_empty_region(x, y):
            reward -= 0.05

        self._prev_score = score
        obs = self._get_obs()
        return obs, reward, done, {"score": score, "step": self._step}

    def render_pil(self) -> Optional[object]:
        if not PIL_OK or self._canvas is None:
            return None
        return Image.fromarray(self._canvas, "RGBA")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Extract 64-float feature vector from current canvas."""
        if self._canvas is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        canvas = self._canvas.astype(np.float32) / 255.0
        rgb    = canvas[:, :, :3]
        alpha  = canvas[:, :, 3]

        # 48: color histogram (16 bins × 3 channels)
        hist_feats = []
        for c in range(3):
            hist, _ = np.histogram(rgb[:, :, c], bins=16, range=(0, 1))
            hist_feats.append(hist / max(hist.sum(), 1))
        hist_arr = np.concatenate(hist_feats)   # 48

        # 8: spatial coverage (2×2×2 grid quadrants — filled vs empty)
        h, w = self.canvas_size, self.canvas_size
        quads = []
        for qi in range(2):
            for qj in range(2):
                region = alpha[qi*h//2:(qi+1)*h//2, qj*w//2:(qj+1)*w//2]
                quads.append(float(region.mean()))
        quad_arr = np.array(quads * 2, dtype=np.float32)   # 8

        # 4: brightness / saturation stats
        brightness = rgb.mean(axis=2)
        bright_feats = np.array([
            float(brightness.mean()),
            float(brightness.std()),
            float(alpha.mean()),
            float(alpha.std()),
        ], dtype=np.float32)   # 4

        # 4: edge density (Sobel-lite)
        if brightness.shape[0] > 2:
            dx = np.abs(np.diff(brightness, axis=1)).mean()
            dy = np.abs(np.diff(brightness, axis=0)).mean()
        else:
            dx = dy = 0.0
        edge_feats = np.array([dx, dy, dx * dy, (dx + dy) / 2], dtype=np.float32)  # 4

        obs = np.concatenate([hist_arr, quad_arr, bright_feats, edge_feats])
        assert obs.shape[0] == self.obs_dim, f"obs dim mismatch: {obs.shape[0]} vs {self.obs_dim}"
        return obs.astype(np.float32)

    def _draw_pixel(self, x: int, y: int, color: tuple, brush: int):
        if self._canvas is None:
            return
        r = brush // 2
        h, w = self._canvas.shape[:2]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    self._canvas[ny, nx] = (*color, 255)

    def _next_position(self) -> Tuple[int, int]:
        """Spiral through grid positions."""
        # 8×8 grid on canvas
        grid_n = self.canvas_size // 4
        total  = grid_n * grid_n
        idx    = self._draw_pos_idx % total
        gx     = (idx % grid_n) * 4 + 2
        gy     = (idx // grid_n) * 4 + 2
        self._draw_pos_idx += 1
        return gx, gy

    def _is_empty_region(self, x: int, y: int) -> bool:
        if self._canvas is None:
            return True
        r = 4
        h, w = self._canvas.shape[:2]
        region = self._canvas[
            max(0, y-r):min(h, y+r),
            max(0, x-r):min(w, x+r),
            3
        ]
        return float(region.mean()) < 10.0

    def _compute_score(self) -> float:
        """Score current canvas state 0-100."""
        if self._score_fn is not None:
            pil_img = self.render_pil()
            return float(self._score_fn(pil_img)) if pil_img else 0.0

        # Fallback heuristic: reward coverage + color variety
        if self._canvas is None:
            return 0.0
        alpha  = self._canvas[:, :, 3]
        coverage = float(alpha.mean()) / 255.0 * 60.0
        n_colors = len(set(map(tuple, self._canvas.reshape(-1, 4)[:, :3].tolist())))
        variety  = min(40.0, n_colors * 1.5)
        return coverage + variety
