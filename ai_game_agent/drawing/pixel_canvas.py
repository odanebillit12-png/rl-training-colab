"""
Pixel Canvas — AI Dev's drawing surface.

AI Dev draws here pixel-by-pixel, just like a human artist with a pencil on a blank canvas.
Supports layers (background / midground / foreground) for 2.75D depth illusion.
No anti-aliasing: true pixel art only.
"""
from __future__ import annotations
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import colorsys
import json


class PixelCanvas:
    """A blank pixel art canvas with every tool an artist needs."""

    LAYERS = ["background", "midground", "foreground"]

    def __init__(self, width: int, height: int, bg_color: str = "#0d0d1a"):
        self.width = width
        self.height = height
        self._current_layer = "background"
        self._layers: Dict[str, Image.Image] = {}
        self.stroke_log: List[Dict] = []          # every command AI issued
        self.color_log: List[str] = []            # every color AI used
        self._init_layers(bg_color)

    # ── Layer management ────────────────────────────────────────────────────

    def _init_layers(self, bg_color: str) -> None:
        for name in self.LAYERS:
            if name == "background":
                self._layers[name] = Image.new(
                    "RGBA", (self.width, self.height), self._parse_color(bg_color)
                )
            else:
                self._layers[name] = Image.new(
                    "RGBA", (self.width, self.height), (0, 0, 0, 0)
                )

    def _layer(self, name: Optional[str] = None) -> Image.Image:
        name = name or self._current_layer
        return self._layers.get(name, self._layers["background"])

    # ── Color helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_color(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        if len(h) != 6:
            return (0, 0, 0, 255)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (r, g, b, alpha)

    @staticmethod
    def _darken(color: Tuple, factor: float) -> Tuple:
        r, g, b, a = color
        return (max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor)), a)

    @staticmethod
    def _lighten(color: Tuple, factor: float) -> Tuple:
        r, g, b, a = color
        return (min(255, int(r + (255 - r) * factor)),
                min(255, int(g + (255 - g) * factor)),
                min(255, int(b + (255 - b) * factor)), a)

    # ── Command executor ────────────────────────────────────────────────────

    def execute(self, commands: List[Dict[str, Any]]) -> int:
        """Execute AI drawing commands. Returns count of successful commands."""
        ok = 0
        for cmd in commands:
            try:
                self._exec_one(cmd)
                self.stroke_log.append(cmd)
                # Track colors for palette analysis
                for key in ("color", "c1", "c2", "outline_color", "fill_color"):
                    if key in cmd and isinstance(cmd[key], str) and cmd[key].startswith("#"):
                        self.color_log.append(cmd[key])
                ok += 1
            except Exception:
                pass  # Bad command — skip without crashing
        return ok

    def _exec_one(self, cmd: Dict[str, Any]) -> None:  # noqa: C901
        name = cmd.get("cmd", "")
        layer_name = cmd.get("layer", self._current_layer)
        img = self._layer(layer_name)
        draw = ImageDraw.Draw(img)

        # ── Meta commands ──
        if name == "layer":
            lname = cmd.get("name", "background")
            if lname in self.LAYERS:
                self._current_layer = lname
            return

        if name == "set_bg":
            c = self._parse_color(cmd["color"])
            img.paste(Image.new("RGBA", (self.width, self.height), c))
            return

        if name == "clear_layer":
            self._layers[layer_name] = Image.new(
                "RGBA", (self.width, self.height),
                (0, 0, 0, 0) if layer_name != "background" else (0, 0, 0, 255)
            )
            return

        # ── Single pixel (pencil) ──
        if name == "pixel":
            x, y = int(cmd["x"]), int(cmd["y"])
            if 0 <= x < self.width and 0 <= y < self.height:
                draw.point((x, y), fill=self._parse_color(cmd["color"]))

        # ── Batch pixels ──
        elif name == "pixels":
            c = self._parse_color(cmd["color"])
            for pt in cmd.get("points", []):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    draw.point((x, y), fill=c)

        # ── Line stroke (pencil drag) ──
        elif name == "line":
            c = self._parse_color(cmd["color"])
            draw.line(
                [(int(cmd["x1"]), int(cmd["y1"])), (int(cmd["x2"]), int(cmd["y2"]))],
                fill=c, width=1
            )

        # ── Rectangle (filled or outline) ──
        elif name == "rect":
            c = self._parse_color(cmd["color"])
            x, y, w, h = int(cmd["x"]), int(cmd["y"]), int(cmd["w"]), int(cmd["h"])
            if cmd.get("fill", True):
                draw.rectangle([x, y, x + w - 1, y + h - 1], fill=c)
            if cmd.get("outline"):
                oc = self._parse_color(cmd.get("outline_color", cmd["color"]))
                draw.rectangle([x, y, x + w - 1, y + h - 1], outline=oc)

        # ── Circle / Ellipse ──
        elif name in ("circle", "ellipse"):
            c = self._parse_color(cmd["color"])
            cx = int(cmd.get("cx", cmd.get("x", self.width // 2)))
            cy = int(cmd.get("cy", cmd.get("y", self.height // 2)))
            rx = int(cmd.get("rx", cmd.get("r", 5)))
            ry = int(cmd.get("ry", cmd.get("r", 5)))
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
            if cmd.get("fill", True):
                draw.ellipse(bbox, fill=c)
            else:
                draw.ellipse(bbox, outline=c)

        # ── Vertical gradient (sky, fog, depth) ──
        elif name == "gradient":
            c1 = self._parse_color(cmd["c1"])
            c2 = self._parse_color(cmd["c2"])
            x1 = int(cmd.get("x1", 0))
            y1 = int(cmd.get("y1", 0))
            x2 = int(cmd.get("x2", self.width))
            y2 = int(cmd.get("y2", self.height))
            span = max(y2 - y1, 1)
            for gy in range(y1, min(y2, self.height)):
                t = (gy - y1) / span
                r = int(c1[0] + (c2[0] - c1[0]) * t)
                g = int(c1[1] + (c2[1] - c1[1]) * t)
                b = int(c1[2] + (c2[2] - c1[2]) * t)
                draw.line(
                    [(x1, gy), (min(x2, self.width - 1), gy)],
                    fill=(r, g, b, 255)
                )

        # ── Pixel-art dither between two colors ──
        elif name == "dither":
            c1 = self._parse_color(cmd["c1"])
            c2 = self._parse_color(cmd["c2"])
            x1, y1 = int(cmd.get("x1", 0)), int(cmd.get("y1", 0))
            x2, y2 = int(cmd.get("x2", self.width)), int(cmd.get("y2", self.height))
            for dy in range(y1, min(y2, self.height)):
                for dx in range(x1, min(x2, self.width)):
                    draw.point((dx, dy), fill=c1 if (dx + dy) % 2 == 0 else c2)

        # ── Shade existing pixels (add shadow / highlight) ──
        elif name == "shade":
            x, y = int(cmd["x"]), int(cmd["y"])
            factor = float(cmd.get("factor", 0.65))
            if 0 <= x < self.width and 0 <= y < self.height:
                px = img.getpixel((x, y))
                new_px = (
                    max(0, int(px[0] * factor)),
                    max(0, int(px[1] * factor)),
                    max(0, int(px[2] * factor)),
                    px[3] if len(px) > 3 else 255,
                )
                draw.point((x, y), fill=new_px)

        # ── Shade a rectangular region ──
        elif name == "shade_rect":
            factor = float(cmd.get("factor", 0.7))
            x, y, w, h = int(cmd["x"]), int(cmd["y"]), int(cmd["w"]), int(cmd["h"])
            for sy in range(y, min(y + h, self.height)):
                for sx in range(x, min(x + w, self.width)):
                    px = img.getpixel((sx, sy))
                    draw.point((sx, sy), fill=(
                        max(0, int(px[0] * factor)),
                        max(0, int(px[1] * factor)),
                        max(0, int(px[2] * factor)),
                        px[3] if len(px) > 3 else 255,
                    ))

        # ── Highlight a rectangular region ──
        elif name == "highlight_rect":
            factor = float(cmd.get("factor", 0.35))
            x, y, w, h = int(cmd["x"]), int(cmd["y"]), int(cmd["w"]), int(cmd["h"])
            for hy_y in range(y, min(y + h, self.height)):
                for hx in range(x, min(x + w, self.width)):
                    px = img.getpixel((hx, hy_y))
                    draw.point((hx, hy_y), fill=(
                        min(255, int(px[0] + (255 - px[0]) * factor)),
                        min(255, int(px[1] + (255 - px[1]) * factor)),
                        min(255, int(px[2] + (255 - px[2]) * factor)),
                        px[3] if len(px) > 3 else 255,
                    ))

        # ── Flood fill (paint bucket) ──
        elif name == "flood_fill":
            x, y = int(cmd["x"]), int(cmd["y"])
            if 0 <= x < self.width and 0 <= y < self.height:
                self._flood_fill(img, x, y, self._parse_color(cmd["color"]))

        # ── Outline pass (1px black/color border around non-transparent pixels) ──
        elif name == "add_outline":
            oc = self._parse_color(cmd.get("color", "#000000"))
            self._add_outline(img, draw, oc)

    # ── Drawing utilities ────────────────────────────────────────────────────

    def _flood_fill(self, img: Image.Image, x: int, y: int, new_color: Tuple) -> None:
        target = img.getpixel((x, y))
        if target == new_color:
            return
        pixels = img.load()
        stack = [(x, y)]
        visited: set = set()
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            if not (0 <= cx < self.width and 0 <= cy < self.height):
                continue
            if pixels[cx, cy] != target:
                continue
            visited.add((cx, cy))
            pixels[cx, cy] = new_color
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

    def _add_outline(self, img: Image.Image, draw: ImageDraw.ImageDraw, color: Tuple) -> None:
        """Draw 1px outline around all non-transparent/non-background pixels."""
        arr = img.load()
        bg = arr[0, 0] if img.mode == "RGBA" else None
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if arr[x, y] == bg:
                    continue
                # If any neighbor is background → it's an edge
                for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                    if arr[nx, ny] == bg:
                        draw.point((nx, ny), fill=color)
                        break

    # ── Composite & save ────────────────────────────────────────────────────

    def composite(self) -> Image.Image:
        """Flatten all layers into final RGB image (what the player sees)."""
        result = self._layers["background"].copy()
        for name in ["midground", "foreground"]:
            result = Image.alpha_composite(result, self._layers[name])
        return result.convert("RGB")

    def save(self, path: str) -> str:
        """Save composited image to disk."""
        img = self.composite()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "PNG")
        return path

    def save_with_layers(self, base_path: str) -> Dict[str, str]:
        """Save final + each layer separately so user can review the depth."""
        paths: Dict[str, str] = {}
        # Composite
        self.save(base_path)
        paths["final"] = base_path
        # Layers
        for name, img in self._layers.items():
            p = base_path.replace(".png", f"_{name}.png")
            img.save(p, "PNG")
            paths[name] = p
        return paths

    def unique_colors(self) -> List[str]:
        """Return unique hex colors used in this drawing."""
        seen = set()
        result = []
        for c in self.color_log:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result

    def stroke_count(self) -> int:
        return len(self.stroke_log)
