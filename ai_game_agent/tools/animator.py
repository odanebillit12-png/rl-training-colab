"""
Sprite Animator — generates frame-by-frame pixel art animations.
Creates walk, run, idle, attack, death, jump cycles.
Exports sprite sheets (PNG) + Godot-ready JSON metadata.
"""
from __future__ import annotations
import math, json
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw
import numpy as np
from ai_game_agent.tools.pixel_artist import CharacterDrawer, quantize_image, draw_to_base64


# ─── Animation definitions ────────────────────────────────────────────────────

ANIMATIONS = {
    "idle":   {"frames": 4,  "fps": 4,  "desc": "Subtle breathing bob"},
    "walk":   {"frames": 8,  "fps": 8,  "desc": "4-directional walk cycle"},
    "run":    {"frames": 6,  "fps": 10, "desc": "Fast run cycle"},
    "attack": {"frames": 6,  "fps": 10, "desc": "Melee swing"},
    "cast":   {"frames": 8,  "fps": 8,  "desc": "Magic casting animation"},
    "death":  {"frames": 5,  "fps": 6,  "desc": "Fall and fade"},
    "jump":   {"frames": 5,  "fps": 8,  "desc": "Jump arc"},
    "hurt":   {"frames": 3,  "fps": 8,  "desc": "Hit flash"},
    "sit":    {"frames": 2,  "fps": 2,  "desc": "Sitting idle"},
    "cheer":  {"frames": 4,  "fps": 6,  "desc": "Victory cheer"},
}


class SpriteAnimator:
    """
    Generates animation frames by programmatically transforming base sprites.
    All motion is computed — no external API needed.
    """

    def __init__(self, archetype: str = "warrior", size: int = 32,
                 palette: str = "db32", seed: int = 0):
        self.archetype = archetype
        self.size = size
        self.palette = palette
        self.seed = seed
        self.drawer = CharacterDrawer(archetype, size, palette, seed)

    # ─── Public ───────────────────────────────────────────────────────────────

    def animate(self, anim_name: str, direction: str = "south") -> list[Image.Image]:
        """Return a list of PIL frames for the given animation."""
        fn = getattr(self, f"_anim_{anim_name}", self._anim_idle)
        return fn(direction)

    def build_sprite_sheet(
        self,
        animations: Optional[list[str]] = None,
        direction: str = "south",
        output_path: Optional[str] = None,
    ) -> tuple[Image.Image, dict]:
        """Pack all animations into a single sprite sheet PNG + JSON metadata."""
        if animations is None:
            animations = ["idle", "walk", "run", "attack", "death"]

        all_frames: list[tuple[str, list[Image.Image]]] = []
        for name in animations:
            frames = self.animate(name, direction)
            all_frames.append((name, frames))

        total_frames = sum(len(f) for _, f in all_frames)
        cols = 8
        rows = math.ceil(total_frames / cols)
        sheet_w = cols * self.size
        sheet_h = rows * self.size

        sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
        meta: dict = {"frame_size": self.size, "animations": {}}

        idx = 0
        for anim_name, frames in all_frames:
            start = idx
            for frame in frames:
                col = idx % cols
                row = idx // cols
                sheet.paste(frame, (col * self.size, row * self.size))
                idx += 1
            meta["animations"][anim_name] = {
                "start": start,
                "end": idx - 1,
                "fps": ANIMATIONS.get(anim_name, {}).get("fps", 8),
                "loop": anim_name not in ("death", "hurt"),
            }

        if output_path:
            sheet.save(output_path)
            json_path = output_path.replace(".png", ".json")
            Path(json_path).write_text(json.dumps(meta, indent=2))

        return sheet, meta

    def build_all_directions_sheet(
        self,
        animations: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Build sprite sheets for all 4 directions."""
        if animations is None:
            animations = ["idle", "walk", "run", "attack"]
        result = {}
        for direction in ["south", "north", "east", "west"]:
            out_path = None
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                out_path = f"{output_dir}/{self.archetype}_{direction}.png"
            sheet, meta = self.build_sprite_sheet(animations, direction, out_path)
            result[direction] = {"sheet": sheet, "meta": meta}
        return result

    # ─── Animation generators ─────────────────────────────────────────────────

    def _base_frame(self, direction: str) -> Image.Image:
        return self.drawer.draw(direction)

    def _anim_idle(self, direction: str) -> list[Image.Image]:
        """Breathing: tiny vertical bob over 4 frames."""
        frames = []
        base = self._base_frame(direction)
        for i in range(4):
            bob = round(math.sin(i * math.pi / 2) * 1)
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            frame.paste(base, (0, bob))
            frames.append(frame)
        return frames

    def _anim_walk(self, direction: str) -> list[Image.Image]:
        """8-frame walk: leg swing + body bob."""
        base = self._base_frame(direction)
        base_arr = np.array(base)
        frames = []
        for i in range(8):
            t = i / 8.0
            bob = round(abs(math.sin(t * math.pi * 2)) * -1)
            lean = round(math.sin(t * math.pi * 2) * 0.5)
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            shifted = Image.fromarray(base_arr)
            frame.paste(shifted, (lean, bob))
            # Add leg alternation via pixel manipulation
            frame = self._add_leg_swing(frame, i, direction)
            frames.append(frame)
        return frames

    def _anim_run(self, direction: str) -> list[Image.Image]:
        """6-frame run: bigger bob + lean."""
        base = self._base_frame(direction)
        base_arr = np.array(base)
        frames = []
        for i in range(6):
            t = i / 6.0
            bob = round(abs(math.sin(t * math.pi * 2)) * -2)
            lean = round(math.sin(t * math.pi * 2) * 1)
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            frame.paste(Image.fromarray(base_arr), (lean, bob))
            frames.append(frame)
        return frames

    def _anim_attack(self, direction: str) -> list[Image.Image]:
        """6-frame attack swing: wind-up, strike, recovery."""
        base = self._base_frame(direction)
        frames = []
        offsets = [(0,0),(-1,-1),(1,0),(2,1),(1,0),(0,0)]
        for ox, oy in offsets:
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            frame.paste(base, (ox, oy))
            # Flash on strike frame
            if ox == 2:
                frame = self._add_strike_flash(frame)
            frames.append(frame)
        return frames

    def _anim_cast(self, direction: str) -> list[Image.Image]:
        """8-frame magic cast: raise arm, glow effect."""
        base = self._base_frame(direction)
        frames = []
        for i in range(8):
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            bob = round(math.sin(i * math.pi / 4) * -1)
            frame.paste(base, (0, bob))
            if i in (3, 4, 5):
                frame = self._add_magic_glow(frame, i)
            frames.append(frame)
        return frames

    def _anim_death(self, direction: str) -> list[Image.Image]:
        """5-frame death: tilt and fade."""
        base = self._base_frame(direction)
        frames = []
        for i in range(5):
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            alpha = max(0, 255 - i * 50)
            drop = i * 2
            faded = base.copy()
            r, g, b, a = faded.split()
            a = a.point(lambda p: min(p, alpha))
            faded = Image.merge("RGBA", (r, g, b, a))
            frame.paste(faded, (0, drop))
            frames.append(frame)
        return frames

    def _anim_jump(self, direction: str) -> list[Image.Image]:
        """5-frame jump arc."""
        base = self._base_frame(direction)
        arc = [0, -4, -6, -4, 0]
        frames = []
        for dy in arc:
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            frame.paste(base, (0, dy))
            frames.append(frame)
        return frames

    def _anim_hurt(self, direction: str) -> list[Image.Image]:
        """3-frame hurt flash: white tint."""
        base = self._base_frame(direction)
        frames = []
        for i in range(3):
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            if i == 1:
                tinted = self._apply_tint(base, (255, 80, 80, 120))
                frame.paste(tinted, (1 if i == 1 else 0, 0))
            else:
                frame.paste(base, (0, 0))
            frames.append(frame)
        return frames

    def _anim_sit(self, direction: str) -> list[Image.Image]:
        base = self._base_frame(direction)
        frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        frame.paste(base, (0, self.size // 8))
        return [frame, frame]

    def _anim_cheer(self, direction: str) -> list[Image.Image]:
        base = self._base_frame(direction)
        frames = []
        for i in range(4):
            bob = -2 if i % 2 == 0 else 0
            frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            frame.paste(base, (0, bob))
            frames.append(frame)
        return frames

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _add_leg_swing(self, frame: Image.Image, step: int, direction: str) -> Image.Image:
        """Shift bottom pixels to simulate leg alternation."""
        arr = np.array(frame)
        s = self.size
        leg_region = arr[s*3//4:, :].copy()
        swing = round(math.sin(step * math.pi / 4) * 2)
        if swing > 0:
            arr[s*3//4:, swing:] = leg_region[:, :-swing] if swing < s else leg_region
        elif swing < 0:
            arr[s*3//4:, :swing] = leg_region[:, -swing:] if -swing < s else leg_region
        return Image.fromarray(arr.astype(np.uint8), "RGBA")

    def _add_strike_flash(self, frame: Image.Image) -> Image.Image:
        d = ImageDraw.Draw(frame)
        s = self.size
        d.ellipse([s//2, s//3, s-2, s*2//3], fill=(255, 240, 100, 160))
        return frame

    def _add_magic_glow(self, frame: Image.Image, phase: int) -> Image.Image:
        d = ImageDraw.Draw(frame)
        s = self.size
        radius = 4 + phase
        cx, cy = s*3//4, s//4
        d.ellipse([cx-radius, cy-radius, cx+radius, cy+radius],
                  fill=(150, 100, 255, 120))
        return frame

    def _apply_tint(self, img: Image.Image, tint: tuple) -> Image.Image:
        tint_layer = Image.new("RGBA", img.size, tint)
        return Image.alpha_composite(img, tint_layer)


# ─── Public helpers ───────────────────────────────────────────────────────────

def generate_character_sheet(
    archetype: str = "warrior",
    size: int = 32,
    animations: Optional[list[str]] = None,
    all_directions: bool = True,
    output_dir: Optional[str] = None,
    palette: str = "db32",
    seed: int = 0,
) -> dict:
    """
    Generate complete sprite sheets for a character.
    Returns dict with sheets per direction and Godot-ready metadata.
    """
    anim = SpriteAnimator(archetype, size, palette, seed)

    if animations is None:
        animations = list(ANIMATIONS.keys())

    if all_directions:
        result = anim.build_all_directions_sheet(animations, output_dir)
    else:
        sheet, meta = anim.build_sprite_sheet(animations, "south", 
            f"{output_dir}/{archetype}_south.png" if output_dir else None)
        result = {"south": {"sheet": sheet, "meta": meta}}

    # Build Godot AnimationPlayer resource info
    godot_meta = _build_godot_meta(archetype, size, result)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/{archetype}_godot.json").write_text(
            json.dumps(godot_meta, indent=2))

    return {"sheets": result, "godot_meta": godot_meta}


def _build_godot_meta(archetype: str, size: int, sheets: dict) -> dict:
    """Build Godot 4 AnimationPlayer resource description."""
    meta = {
        "character": archetype,
        "frame_size": size,
        "directions": {},
        "godot_setup": {
            "node": "AnimatedSprite2D",
            "sprite_frames": f"{archetype}_frames.tres",
            "instructions": [
                f"1. Import each {archetype}_<direction>.png as a texture in Godot",
                "2. Create AnimatedSprite2D node",
                "3. Create new SpriteFrames resource",
                "4. For each animation: add frames from the sprite sheet using the JSON metadata",
                "5. Set the fps from metadata per animation",
            ]
        }
    }
    for direction, data in sheets.items():
        meta["directions"][direction] = data.get("meta", {})
    return meta


def frames_to_base64_list(frames: list[Image.Image]) -> list[str]:
    return [draw_to_base64(f) for f in frames]
