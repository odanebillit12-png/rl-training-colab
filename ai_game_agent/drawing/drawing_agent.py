"""
Drawing Agent — AI Dev's brain for making pixel art.

AI Dev STUDIES reference images, then DRAWS on a blank canvas pixel-by-pixel.
Uses Gemini to generate JSON drawing commands. Gets penalized for bad art,
rewarded for quality matching GOTY-level standards.

The AI learns by:
  1. Studying reference images (technique memory)
  2. Planning the drawing (layers, palette, composition)
  3. Issuing pixel-level commands on blank canvas
  4. Reviewing its own work
  5. Improving from penalty/reward feedback
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pixel_canvas import PixelCanvas
from .reference_library import ReferenceLibrary
from .pixel_art_researcher import PixelArtResearcher


DRAWING_SYSTEM_PROMPT = """You are AI Dev, a pixel art master learning to draw at GOTY (Game of the Year) level.

You draw pixel art from scratch on a blank canvas using ONLY drawing commands.
You have studied many reference images and learned their techniques.
You MUST apply those techniques precisely.

YOUR GOAL: Create pixel art so polished it looks like it belongs in Celeste, 
Hollow Knight, Sea of Stars, or Dead Cells — 2.75D depth, perfect color harmony, 
clean silhouettes, professional shading.

DRAWING COMMANDS YOU CAN USE:
- {"cmd":"layer","name":"background"|"midground"|"foreground"}  ← switch layer
- {"cmd":"set_bg","color":"#hex"}                              ← fill canvas background
- {"cmd":"gradient","c1":"#hex","c2":"#hex","y1":0,"y2":32}   ← gradient fill area
- {"cmd":"dither","c1":"#hex","c2":"#hex","x1":0,"y1":0,"x2":16,"y2":8} ← pixel dither
- {"cmd":"rect","x":0,"y":0,"w":16,"h":8,"color":"#hex","fill":true}
- {"cmd":"pixel","x":5,"y":3,"color":"#hex"}                  ← single pixel (pencil)
- {"cmd":"pixels","points":[[x,y],[x,y]],"color":"#hex"}      ← batch pixels
- {"cmd":"line","x1":0,"y1":0,"x2":15,"y2":0,"color":"#hex"} ← straight line
- {"cmd":"circle","cx":8,"cy":8,"r":4,"color":"#hex","fill":true}
- {"cmd":"ellipse","cx":8,"cy":8,"rx":4,"ry":2,"color":"#hex","fill":false}
- {"cmd":"flood_fill","x":8,"y":8,"color":"#hex"}             ← paint bucket
- {"cmd":"shade","x":5,"y":3,"factor":0.6}                    ← darken one pixel
- {"cmd":"shade_rect","x":0,"y":0,"w":8,"h":4,"factor":0.65} ← shadow region
- {"cmd":"highlight_rect","x":0,"y":0,"w":4,"h":2,"factor":0.3} ← highlight region
- {"cmd":"add_outline","color":"#000000"}                     ← add 1px outline to current layer

DRAWING WORKFLOW:
1. Plan your palette (6-16 colors max for the whole piece)
2. Switch to "background" layer → paint sky/void/atmosphere
3. Switch to "midground" layer → paint main terrain/structures  
4. Switch to "foreground" layer → paint detail/props/characters
5. Add shadows and highlights (shade_rect / highlight_rect)
6. Add outline pass if needed (add_outline)

2.75D RULES YOU MUST FOLLOW:
- Background: desaturated, blue-shifted, sparse detail
- Midground: normal saturation, medium detail
- Foreground: maximum saturation, maximum detail, darkest values at base
- Every object casts a shadow consistent with top-left light source
- Limit pixel size to EXACTLY 1px — no 2px blocks unless intentional
- Silhouettes must be readable at thumbnail size

PALETTE RULES:
- Choose a cohesive palette of 8-16 colors BEFORE drawing
- For each color: shadow = color × 0.6, highlight = color + 40% white
- Background palette: same hues but 40% less saturated
- Never use pure white (#ffffff) for highlights — use near-white (#f0f0e8)
- Never use pure black (#000000) for shadows — use deep dark (#0d0d1a or #1a0d2e)
"""


class RateLimitError(RuntimeError):
    """Raised when ALL LLM backends are currently rate-limited."""


class DrawingAgent:
    """AI Dev — learns from references, then draws from scratch."""

    def __init__(
        self,
        api_key: str,                            # Gemini API key
        anthropic_key: str = "",                 # Claude API key (primary if set)
        github_key: str = "",                    # GitHub Models PAT(s) — comma-separated list
        groq_key: str = "",                      # Groq free-tier key (14,400 req/day)
        brave_key: str = "",                     # Brave Search API key (2,000 free/month)
        ref_library: Optional[ReferenceLibrary] = None,
        memory_file: str = "training_data/drawing_memory.json",
    ):
        self.api_key = api_key
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.groq_key = groq_key or os.getenv("GROQ_API_KEY", "")

        # Support multiple GitHub PATs for rotation (comma-separated or list)
        raw_keys = github_key or os.getenv("GITHUB_TOKEN", "")
        if isinstance(raw_keys, str):
            self._github_keys: List[str] = [k.strip() for k in raw_keys.split(",") if k.strip()]
        else:
            self._github_keys = list(raw_keys)
        self._github_key_idx: int = 0   # current PAT index

        self.ref_library = ref_library or ReferenceLibrary()
        self.researcher = PixelArtResearcher(brave_key=brave_key or os.getenv("BRAVE_API_KEY", ""))
        self.memory_file = Path(memory_file)
        self._memory: List[Dict] = []
        self._technique_notes: str = ""
        self._load_memory()
        self._compile_techniques()

    @property
    def github_key(self) -> str:
        """Current active GitHub PAT."""
        if not self._github_keys:
            return ""
        return self._github_keys[self._github_key_idx % len(self._github_keys)]

    def _rotate_github_key(self) -> bool:
        """Rotate to next GitHub PAT. Returns True if a new key is available."""
        if len(self._github_keys) <= 1:
            return False
        next_idx = (self._github_key_idx + 1) % len(self._github_keys)
        if next_idx == self._github_key_idx:
            return False
        self._github_key_idx = next_idx
        print(f"   🔄 Rotating to GitHub PAT #{self._github_key_idx + 1}/{len(self._github_keys)}")
        return True

    def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call LLM backends in priority order: Claude → GitHub Models (with rotation) → Gemini."""
        # 1. Claude (Anthropic) — best structured JSON output
        if self.anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
                msg = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text
            except Exception as e:
                err = str(e)
                if "credit" in err.lower() or "billing" in err.lower() or "balance" in err.lower():
                    pass  # out of credits — skip silently
                else:
                    print(f"   ℹ️  Claude unavailable: {err[:60]}, trying GitHub Models...")

        # 2. GitHub Models (GPT-4o-mini) — rotate through all PATs on 429
        if self._github_keys:
            from openai import OpenAI
            keys_tried = 0
            while keys_tried < len(self._github_keys):
                try:
                    client = OpenAI(
                        base_url="https://models.inference.ai.azure.com",
                        api_key=self.github_key,
                    )
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp.choices[0].message.content
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RateLimitReached" in err or "rate_limit" in err.lower():
                        keys_tried += 1
                        if not self._rotate_github_key() or keys_tried >= len(self._github_keys):
                            print(f"   ⚠️  All {len(self._github_keys)} GitHub PAT(s) rate-limited")
                            break
                    else:
                        print(f"   ℹ️  GitHub Models error: {err[:60]}, trying Gemini...")
                        break

        # 3. Groq (free tier) — Llama 3.3 70B, 14,400 req/day
        # Retry up to 3 times with a short wait — Groq resets per minute
        if self.groq_key:
            for groq_attempt in range(3):
                try:
                    return self._call_groq(prompt, max_tokens)
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate_limit" in err.lower():
                        if groq_attempt < 2:
                            wait = (groq_attempt + 1) * 8  # 8s, 16s
                            print(f"   ⚠️  Groq rate-limited — retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            print("   ⚠️  Groq rate-limited, trying Gemini...")
                    else:
                        print(f"   ℹ️  Groq error: {err[:60]}, trying Gemini...")
                        break

        # 4. Gemini (Google) — new SDK, one quick attempt per model (no long waits)
        if self.api_key:
            try:
                from google import genai as gai
                client = gai.Client(api_key=self.api_key)
                for model_name in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
                    try:
                        resp = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                        return resp.text
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "quota" in err.lower() or "Resource has been exhausted" in err:
                            continue
                        break
            except ImportError:
                pass  # new SDK not installed, skip

        # All backends exhausted — raise so trainer can skip the episode
        raise RateLimitError("All LLM backends rate-limited — skipping episode")

    def _call_groq(self, prompt: str, max_tokens: int = 4096) -> str:
        """Groq free-tier: llama-3.3-70b. 14,400 req/day — very generous."""
        from groq import Groq
        client = Groq(api_key=self.groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    # ── Memory ────────────────────────────────────────────────────────────

    def _load_memory(self) -> None:
        if self.memory_file.exists():
            try:
                self._memory = json.loads(self.memory_file.read_text())
            except Exception:
                self._memory = []

    def _save_memory(self) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(json.dumps(self._memory[-200:], indent=2))

    def _compile_techniques(self) -> None:
        """Compile all learned techniques into a summary string."""
        self._technique_notes = self.ref_library.lesson_summary()

    def add_episode(self, episode: Dict) -> None:
        """Record a completed drawing episode for future learning."""
        self._memory.append(episode)
        self._save_memory()

    def recent_feedback(self, n: int = 6) -> str:
        """Format recent episode scores + feedback for prompt injection."""
        if not self._memory:
            return "No previous episodes yet — this is your first drawing."
        recent = self._memory[-n:]
        lines = ["Recent drawing performance:"]
        for ep in recent:
            score = ep.get("score", 0)
            task = ep.get("task", "unknown")
            fb = ep.get("feedback", "")
            icon = "✅" if score >= 75 else ("⚠️" if score >= 55 else "❌")
            lines.append(f"  {icon} {task}: {score}/100 — {fb[:80]}")
        return "\n".join(lines)

    # ── Drawing plan generation ───────────────────────────────────────────

    def generate_drawing_plan(self, task: Dict[str, Any], reward_prompt: str = "") -> str:
        """Plan the drawing before issuing commands."""
        try:
            w, h = task.get("width", 64), task.get("height", 64)
            subject = task.get("description", "a pixel art scene")
            style = task.get("style", "2.75D RPG")

            motivation_block = ""
            if reward_prompt:
                motivation_block = f"\n🏆 COACH SAYS: {reward_prompt}\n"

            # Research visual references before planning
            reference_context = self.researcher.research(task)

            prompt = f"""You are AI Dev planning a pixel art drawing.

Canvas: {w}×{h} pixels
Subject: {subject}
Style: {style}
{motivation_block}
=== VISUAL REFERENCE CONTEXT ===
{reference_context}

=== LEARNED TECHNIQUES ===
{self._technique_notes[:1800]}

{self.recent_feedback()}

Plan your drawing in 3-5 sentences:
1. What palette will you use? (list 8-12 specific hex colors)
2. What goes on each layer (background / midground / foreground)?
3. Where is the light source and how will you show depth?
4. What makes this piece GOTY quality?

Keep your plan under 200 words. Be specific with hex colors."""

            return self._call_llm(prompt, max_tokens=512)
        except RateLimitError:
            raise  # propagate to trainer
        except Exception as e:
            return f"Plan: Draw {task.get('description','scene')} with 2.75D depth, top-left light, 12 colors."

    # ── Drawing command generation ────────────────────────────────────────

    def generate_drawing_commands(
        self, task: Dict[str, Any], plan: str, penalty_hint: str = ""
    ) -> Tuple[List[Dict], str]:
        """Generate drawing commands for the canvas. Returns (commands, raw)."""
        try:
            w, h = task.get("width", 64), task.get("height", 64)
            subject = task.get("description", "a pixel art scene")
            style = task.get("style", "2.75D RPG")
            max_cmds = task.get("max_commands", 400)

            penalty_section = ""
            if penalty_hint:
                penalty_section = f"\n⚠️ PREVIOUS ATTEMPT PENALTY: {penalty_hint}\nFix these issues this time!\n"

            prompt = f"""{DRAWING_SYSTEM_PROMPT}

=== YOUR DRAWING TASK ===
Canvas: {w}×{h} pixels (all coordinates must be within 0-{w-1} x, 0-{h-1} y)
Subject: {subject}
Style: {style}
Max commands: {max_cmds}

=== YOUR PLAN ===
{plan}

=== TECHNIQUE MEMORY ===
{self._technique_notes[:1500]}

=== FEEDBACK FROM LAST SESSION ===
{self.recent_feedback(4)}
{penalty_section}

=== NOW DRAW ===
Output a JSON array of drawing commands ONLY. No explanation. Wrap in ```json ... ``` or output the array directly.
Draw on ALL 3 LAYERS. Use at least 80 commands. Add proper shadows and highlights.
Make it look like {subject} belongs in a AAA pixel art game."""

            raw = self._call_llm(prompt, max_tokens=6000)
            commands = self._parse_commands(raw)
            return commands, raw

        except RateLimitError:
            raise  # propagate — trainer will skip this episode cleanly
        except Exception as e:
            print(f"  ⚠️ DrawingAgent error: {e}")
            return self._fallback_commands(task), ""

    def _parse_commands(self, raw: str) -> List[Dict]:
        """Extract JSON array from Gemini response."""
        # Strip markdown fences
        text = raw
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("["):
                    text = p
                    break

        # Find first [ ... ] block
        start = text.find("[")
        if start == -1:
            return []
        # Find matching closing bracket
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            end = len(text)

        try:
            commands = json.loads(text[start:end])
            if isinstance(commands, list):
                return [c for c in commands if isinstance(c, dict) and "cmd" in c]
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed = self._fix_json(text[start:end])
            try:
                commands = json.loads(fixed)
                if isinstance(commands, list):
                    return [c for c in commands if isinstance(c, dict) and "cmd" in c]
            except Exception:
                pass
        return []

    def _fix_json(self, raw: str) -> str:
        """Fix common JSON issues from LLM output."""
        # Remove trailing commas before ] or }
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        # Fix single quotes
        raw = raw.replace("'", '"')
        return raw

    def _fallback_commands(self, task: Dict) -> List[Dict]:
        """Minimal fallback drawing if Gemini fails."""
        w, h = task.get("width", 64), task.get("height", 64)
        return [
            {"cmd": "layer", "name": "background"},
            {"cmd": "gradient", "c1": "#1a1a3e", "c2": "#0d1a0d", "y1": 0, "y2": h},
            {"cmd": "layer", "name": "midground"},
            {"cmd": "rect", "x": 0, "y": h//2, "w": w, "h": h//4, "color": "#2d4a22", "fill": True},
            {"cmd": "layer", "name": "foreground"},
            {"cmd": "rect", "x": 0, "y": h*3//4, "w": w, "h": h//4, "color": "#1a2e11", "fill": True},
        ]

    # ── Self-critique ─────────────────────────────────────────────────────

    def self_critique(self, score: float, breakdown: Dict, task: Dict) -> str:
        """AI Dev critiques its own work to improve next time."""
        try:
            prompt = f"""You are AI Dev reviewing your own pixel art drawing.

Task: {task.get('description', 'scene')}
Score: {score}/100
Score breakdown: {json.dumps(breakdown, indent=2)}

In 2-3 sentences, identify:
1. The BIGGEST weakness in this score
2. ONE specific change to make next time (be concrete — mention colors, layer usage, etc.)

Be harsh. GOTY standard requires 90+."""
            return self._call_llm(prompt, max_tokens=256).strip()[:300]
        except Exception:
            weak = min(breakdown, key=breakdown.get) if breakdown else "overall"
            return f"Weakest area: {weak} ({breakdown.get(weak, 0)}/100). Improve depth layering next."

    def full_draw_session(
        self,
        task: Dict[str, Any],
        penalty_hint: str = "",
        reward_prompt: str = "",
        save_path: Optional[str] = None,
    ) -> Tuple[PixelCanvas, List[Dict], str]:
        """
        Complete drawing session: plan → draw → execute on canvas.
        Returns (canvas, commands, plan).
        """
        w, h = task.get("width", 64), task.get("height", 64)
        bg = task.get("bg_color", "#0d0d1a")

        # Create blank canvas
        canvas = PixelCanvas(w, h, bg)

        # Plan (inject reward context)
        plan = self.generate_drawing_plan(task, reward_prompt=reward_prompt)

        # Generate commands (inject penalty)
        commands, raw = self.generate_drawing_commands(task, plan, penalty_hint)

        # Execute pixel-by-pixel
        executed = canvas.execute(commands)
        print(f"   🖊️  Drew {executed}/{len(commands)} commands on {w}×{h} canvas")

        # Save
        if save_path:
            paths = canvas.save_with_layers(save_path)
            print(f"   💾  Saved: {save_path} (+ {len(paths)-1} layer files)")

        return canvas, commands, plan
