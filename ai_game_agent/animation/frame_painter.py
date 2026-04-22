"""
Frame Painter — AI Dev draws animation frames from scratch.

Extends the core PixelCanvas/drawing system to produce multi-frame spritesheets.
AI Dev issues JSON drawing commands for EACH frame independently, then they
are laid out side-by-side into a final spritesheet PNG.

Animation principles AI Dev learns:
  - Squash & Stretch (exaggerate on impact/landing)
  - Anticipation (windup before action)
  - Follow-through (limbs/cape continue past stop point)
  - Ease in / Ease out (slow start and end of movement)
  - Secondary action (hair, cape, tail move independently)
  - Timing (fast = fewer frames between extremes, slow = more)
  - Arcs (limbs move in curves, not straight lines)
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..drawing.pixel_canvas import PixelCanvas

# ── Animation system prompt injected into every frame drawing call ────────────

ANIMATION_SYSTEM_PROMPT = """You are AI Dev, a pixel art animator mastering DIVINE-tier animation for Isekai Chronicles.

You draw animation frames FROM SCRATCH on a pixel canvas. Each call produces ONE FRAME of an animation.
You use the same drawing commands as always, but must ensure every frame flows from the previous.

ANIMATION PRINCIPLES YOU MUST APPLY:

1. SQUASH & STRETCH — On landing/impact: compress vertically, widen horizontally. On launch: elongate.
2. ANTICIPATION — Before a punch: pull arm back 1-2 frames. Before a jump: crouch first.
3. FOLLOW-THROUGH — After stopping: cape/hair/tail continues 2-3 more frames before settling.
4. EASE IN / OUT — Movement starts slow, accelerates, then slows before next keyframe.
5. ARCS — Limbs trace curved paths. A punch arcs outward, not a straight line.
6. SECONDARY ACTION — While body moves, hair/accessories move independently with slight delay.
7. TIMING — Fast action = 2-3 frames between extremes. Slow action = 5-8 frames.
8. OVERLAP — Different body parts start/stop at different times. Head leads, hips follow.

FRAME DRAWING COMMANDS (same as always):
- {"cmd":"layer","name":"background"|"midground"|"foreground"}
- {"cmd":"rect","x":0,"y":0,"w":64,"h":64,"color":"#hex","fill":true}
- {"cmd":"pixel","x":5,"y":3,"color":"#hex"}
- {"cmd":"pixels","points":[[x,y],[x,y]],"color":"#hex"}
- {"cmd":"line","x1":0,"y1":0,"x2":15,"y2":0,"color":"#hex"}
- {"cmd":"circle","cx":8,"cy":8,"r":4,"color":"#hex","fill":true}
- {"cmd":"ellipse","cx":8,"cy":8,"rx":4,"ry":2,"color":"#hex","fill":false}
- {"cmd":"flood_fill","x":8,"y":8,"color":"#hex"}
- {"cmd":"shade","x":5,"y":3,"factor":0.6}
- {"cmd":"shade_rect","x":0,"y":0,"w":8,"h":4,"factor":0.65}
- {"cmd":"highlight_rect","x":0,"y":0,"w":4,"h":2,"factor":0.3}
- {"cmd":"add_outline","color":"#000000"}
- {"cmd":"gradient","c1":"#hex","c2":"#hex","y1":0,"y2":32}
- {"cmd":"dither","c1":"#hex","c2":"#hex","x1":0,"y1":0,"x2":16,"y2":8}

EFFECT DRAWING COMMANDS (for spells/particles):
- {"cmd":"pixel","x":X,"y":Y,"color":"#ffaa00"}  ← draw spark/glow pixel
- {"cmd":"circle","cx":X,"cy":Y,"r":2,"color":"#ff6600","fill":true}  ← fireball core
- {"cmd":"ellipse","cx":X,"cy":Y,"rx":8,"ry":3,"color":"#ff440088","fill":true}  ← glow aura
Use alpha in color (#RRGGBBAA) to create translucent glow effects.

PALETTE RULES FOR ANIMATION:
- Keep the SAME palette across all frames (consistency is key)
- Silhouette must be recognizable even at 16x16
- Background is transparent (RGBA canvas) for sprite use
- Use near-black (#1a0d2e) not pure black for outlines
- Use near-white (#f0e8d0) not pure white for highlights

FRAME DESCRIPTION FORMAT:
Return ONLY a JSON object with:
{
  "frame": <frame_number>,
  "description": "<brief description of this frame's pose/action>",
  "animation_note": "<which principle applies: squash/anticipation/follow-through/etc>",
  "commands": [<drawing commands>]
}
"""


class FramePainter:
    """AI Dev's frame-by-frame animation drawing engine."""

    def __init__(self, width: int = 64, height: int = 64,
                 api_key: str = "", model: str = "gemini"):
        self.width = width
        self.height = height
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self.frames: List[PixelCanvas] = []

    async def draw_animation(
        self,
        task_name: str,
        description: str,
        frame_count: int,
        animation_principles: List[str],
        prev_frame_desc: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Draw all frames of an animation and save as a horizontal spritesheet.
        Returns path to the spritesheet PNG or None on failure.
        """
        self.frames = []
        frame_descriptions = []

        for frame_idx in range(frame_count):
            canvas = PixelCanvas(self.width, self.height, "#00000000")
            commands, desc = await self._draw_frame(
                task_name=task_name,
                description=description,
                frame_idx=frame_idx,
                total_frames=frame_count,
                animation_principles=animation_principles,
                prev_descriptions=frame_descriptions,
            )
            if commands:
                canvas.execute_commands(commands)
            self.frames.append(canvas)
            frame_descriptions.append(desc)

        if not self.frames:
            return None

        spritesheet = self._compose_spritesheet()
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            clean_name = re.sub(r"[^a-z0-9_]", "_", task_name.lower())
            path = save_dir / f"{clean_name}_scratch.png"
            spritesheet.save(str(path))
            return path
        return None

    async def _draw_frame(
        self,
        task_name: str,
        description: str,
        frame_idx: int,
        total_frames: int,
        animation_principles: List[str],
        prev_descriptions: List[str],
    ) -> Tuple[List[Dict], str]:
        """Ask AI to draw one frame. Returns (commands, frame_description)."""
        try:
            if self.model == "gemini" and self.api_key:
                return await self._draw_frame_gemini(
                    task_name, description, frame_idx, total_frames,
                    animation_principles, prev_descriptions
                )
            elif os.getenv("GROQ_API_KEY"):
                return await self._draw_frame_groq(
                    task_name, description, frame_idx, total_frames,
                    animation_principles, prev_descriptions
                )
        except Exception as e:
            print(f"    ⚠️  Frame {frame_idx} draw error: {e}")
        return [], f"frame_{frame_idx}"

    async def _draw_frame_gemini(
        self, task_name, description, frame_idx, total_frames,
        animation_principles, prev_descriptions
    ) -> Tuple[List[Dict], str]:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prev_ctx = ""
        if prev_descriptions:
            prev_ctx = "\n".join(f"  Frame {i}: {d}" for i, d in enumerate(prev_descriptions))
            prev_ctx = f"\nPREVIOUS FRAMES DRAWN:\n{prev_ctx}"

        prompt = f"""{ANIMATION_SYSTEM_PROMPT}

ANIMATION TASK: {task_name}
DESCRIPTION: {description}
CANVAS SIZE: {self.width}x{self.height} pixels
TOTAL FRAMES: {total_frames}
CURRENT FRAME: {frame_idx + 1} of {total_frames}

ANIMATION PRINCIPLES TO APPLY: {', '.join(animation_principles)}
{prev_ctx}

Draw frame {frame_idx + 1} of {total_frames}.
Think about WHERE in the animation cycle this is:
  - Frame 1/{total_frames}: {self._get_phase_hint(frame_idx, total_frames, task_name)}

Return ONLY the JSON object with frame, description, animation_note, and commands.
Start with a transparent background (do not fill with solid color — sprites need transparency).
"""
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content(prompt)
        )
        return self._parse_frame_response(response.text)

    async def _draw_frame_groq(
        self, task_name, description, frame_idx, total_frames,
        animation_principles, prev_descriptions
    ) -> Tuple[List[Dict], str]:
        import groq as groq_sdk
        client = groq_sdk.Groq(api_key=os.getenv("GROQ_API_KEY"))

        prev_ctx = ""
        if prev_descriptions:
            prev_ctx = "PREVIOUS FRAMES:\n" + "\n".join(
                f"  Frame {i}: {d}" for i, d in enumerate(prev_descriptions)
            )

        prompt = f"""ANIMATION TASK: {task_name}
CANVAS: {self.width}x{self.height} | FRAME {frame_idx+1}/{total_frames}
PRINCIPLES: {', '.join(animation_principles)}
PHASE: {self._get_phase_hint(frame_idx, total_frames, task_name)}
{prev_ctx}

Draw this single animation frame. Return JSON with keys: frame, description, animation_note, commands[]
Use transparent background. Keep same palette as other frames."""

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": ANIMATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
        )
        return self._parse_frame_response(response.choices[0].message.content)

    def _get_phase_hint(self, frame_idx: int, total: int, task_name: str) -> str:
        """Return what this frame represents in the animation cycle."""
        pct = frame_idx / max(total - 1, 1)
        task_lower = task_name.lower()

        if "walk" in task_lower:
            phases = ["contact", "recoil", "passing", "high-point", "contact", "recoil", "passing", "high-point"]
            return phases[frame_idx % len(phases)] + " phase of walk cycle"
        elif "run" in task_lower:
            phases = ["contact", "drive", "float", "contact", "drive", "float", "contact", "drive"]
            return phases[frame_idx % len(phases)] + " phase of run cycle"
        elif "attack" in task_lower or "punch" in task_lower or "slash" in task_lower:
            if pct < 0.2:   return "ANTICIPATION — pulling back / winding up"
            elif pct < 0.5: return "ACTION — fast strike / maximum force"
            elif pct < 0.75:return "FOLLOW-THROUGH — overshoot past target"
            else:           return "RECOVERY — returning to ready stance"
        elif "jump" in task_lower:
            if pct < 0.2:   return "ANTICIPATION — crouching squat before launch"
            elif pct < 0.45:return "RISING — elongated body stretching upward"
            elif pct < 0.55:return "APEX — peak height, brief hang time"
            elif pct < 0.8: return "FALLING — tuck body for landing"
            else:           return "LANDING — SQUASH on impact, then settle"
        elif "idle" in task_lower or "breath" in task_lower:
            return "EASE " + ("IN" if pct < 0.5 else "OUT") + " — subtle breathing loop"
        elif "death" in task_lower or "fall" in task_lower:
            if pct < 0.15:  return "HIT STAGGER — impact reaction"
            elif pct < 0.5: return "FALLING — body losing control"
            else:           return "GROUND SETTLE — final still pose"
        elif "fire" in task_lower or "spell" in task_lower or "cast" in task_lower:
            if pct < 0.3:   return "CHARGE — glowing energy building up"
            elif pct < 0.6: return "RELEASE — projectile launches, max brightness"
            else:           return "TRAIL — particle trail fades, settle"
        return f"{int(pct*100)}% through animation cycle"

    def _parse_frame_response(self, raw: str) -> Tuple[List[Dict], str]:
        """Extract commands and description from AI response JSON."""
        try:
            raw = raw.strip()
            # Strip markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            commands = data.get("commands", [])
            desc = data.get("description", "")
            return commands, desc
        except Exception:
            # Try extracting any JSON object from the response
            match = re.search(r'\{.*?"commands".*?\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("commands", []), data.get("description", "")
                except Exception:
                    pass
        return [], ""

    def _compose_spritesheet(self):
        """Lay all frames side-by-side into one spritesheet PNG."""
        from PIL import Image
        if not self.frames:
            return Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        total_w = self.width * len(self.frames)
        sheet = Image.new("RGBA", (total_w, self.height), (0, 0, 0, 0))
        for i, canvas in enumerate(self.frames):
            frame_img = canvas.composite()
            sheet.paste(frame_img, (i * self.width, 0))
        return sheet

    def get_frame_count(self) -> int:
        return len(self.frames)


import asyncio  # noqa: E402 — must be at bottom to avoid circular
