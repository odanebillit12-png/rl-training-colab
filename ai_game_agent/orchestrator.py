"""
LLM Orchestrator — routes prompts to Claude, GPT-4, or local Ollama.
Maintains conversation context and injects system persona.
"""
from __future__ import annotations
import json, os, re
from typing import Any
from ai_game_agent.config import (
    AI_MODE, LLM_PROVIDER, LLM_MODEL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
    OLLAMA_URL, OLLAMA_MODEL,
)

SYSTEM_PROMPT = """\
You are an expert game developer AI assistant specialising in Godot 4 (GDScript), \
pixel art RPGs, isekai / JRPG worldbuilding, and colony simulation design. \
You have deep knowledge of:
- Godot 4 scene trees, GDScript, TileMaps, CharacterBody2D, AnimationPlayer
- Pixel art RPG conventions (top-down, 2.5D, JRPG, action-RPG)
- Isekai anime tropes: class systems, reincarnation, overpowered skills, fantasy economies
- Colony / sandbox simulation (villager AI, job systems, resource loops)
- PixelLab AI art generation prompts for Wang tilesets, characters, animations

When generating GDScript:
- Always use Godot 4 syntax (not Godot 3)
- Add extends, class_name, @export, @onready where appropriate
- Include docstring comments on public functions
- Keep scripts modular and under 200 lines each

When generating JSON scene data, use the exact schema expected by the scene builder tool.
Always reply with clean, runnable code. No placeholder comments like # TODO.
"""


class LLMOrchestrator:
    def __init__(self):
        self.history: list[dict] = []
        self._client = None

    # ─── Public API ───────────────────────────────────────────────────────────

    def chat(self, user_message: str, context: str = "") -> str:
        """Send a message and return the assistant reply."""
        if context:
            user_message = f"[Context]\n{context}\n\n[Request]\n{user_message}"
        self.history.append({"role": "user", "content": user_message})
        reply = self._call_llm(self.history)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def one_shot(self, prompt: str, context: str = "") -> str:
        """Single call with no history — for code generation tasks."""
        if context:
            prompt = f"[Context]\n{context}\n\n[Task]\n{prompt}"
        return self._call_llm([{"role": "user", "content": prompt}])

    def reset(self):
        self.history = []

    def extract_code_blocks(self, text: str) -> list[dict[str, str]]:
        """Extract ```language\\ncode``` blocks from a reply."""
        pattern = r"```(\w*)\n(.*?)```"
        blocks = []
        for lang, code in re.findall(pattern, text, re.DOTALL):
            blocks.append({"language": lang or "text", "code": code.strip()})
        return blocks

    # ─── Routing ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        mode = AI_MODE
        provider = LLM_PROVIDER
        if mode == "local":
            return self._ollama(messages)
        if provider == "gemini":
            if not GEMINI_API_KEY:
                return self._demo(messages)
            return self._gemini(messages)
        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                return self._demo(messages)
            return self._anthropic(messages)
        if provider == "openai":
            if not OPENAI_API_KEY:
                return self._demo(messages)
            return self._openai(messages)
        return self._ollama(messages)

    # ─── Anthropic ────────────────────────────────────────────────────────────

    def _anthropic(self, messages: list[dict]) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text

    # ─── OpenAI ───────────────────────────────────────────────────────────────

    def _openai(self, messages: list[dict]) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("pip install openai")
        client = OpenAI(api_key=OPENAI_API_KEY)
        full = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        resp = client.chat.completions.create(
            model=LLM_MODEL or "gpt-4o",
            messages=full,
            max_tokens=8192,
        )
        return resp.choices[0].message.content

    # ─── Google Gemini ────────────────────────────────────────────────────────

    def _gemini(self, messages: list[dict]) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("pip install google-genai")
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_name = LLM_MODEL if LLM_MODEL else "gemini-2.0-flash"
        # Build contents list (exclude system — passed via config)
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=8192)
        resp = client.models.generate_content(model=model_name, contents=contents, config=cfg)
        return resp.text

    # ─── Demo mode (no API key) ───────────────────────────────────────────────

    def _demo(self, messages: list[dict]) -> str:
        """Smart template replies when no API key is configured."""
        last = messages[-1]["content"].lower() if messages else ""
        if any(w in last for w in ["walk", "run", "anim"]):
            return (
                "🎮 **Demo Mode** (add ANTHROPIC_API_KEY to .env for full AI).\n\n"
                "Here's a GDScript walk animation setup:\n\n"
                "```gdscript\nextends CharacterBody2D\n\n"
                "@onready var anim = $AnimationPlayer\n\n"
                "func _physics_process(delta):\n"
                "    var dir = Input.get_axis(\"ui_left\", \"ui_right\")\n"
                "    velocity.x = dir * 150\n"
                "    if dir != 0:\n"
                "        anim.play(\"walk\")\n"
                "    else:\n"
                "        anim.play(\"idle\")\n"
                "    move_and_slide()\n```"
            )
        if any(w in last for w in ["tileset", "tile", "map", "world"]):
            return (
                "🎮 **Demo Mode** — Add your API key to unlock full AI.\n\n"
                "For AAA-quality tilemaps in Godot 4:\n"
                "- Use 32×32 Wang tiles with 4-corner autotiling\n"
                "- Layer: Ocean → Beach → Grass → Forest → Mountain\n"
                "- Add a YSort node so characters render behind trees\n"
                "- Use a noise texture (FastNoiseLite) to blend biomes\n\n"
                "PixelLab MCP is connected — I can generate tiles right now! "
                "Just say: 'generate forest tileset' or 'create grass tiles'."
            )
        if any(w in last for w in ["script", "player", "character", "rpg"]):
            return (
                "🎮 **Demo Mode** — Add ANTHROPIC_API_KEY to .env for full generation.\n\n"
                "```gdscript\nextends CharacterBody2D\nclass_name Player\n\n"
                "@export var speed: float = 200.0\n"
                "@export var hp: int = 100\n\n"
                "@onready var sprite = $AnimatedSprite2D\n\n"
                "func _physics_process(delta: float) -> void:\n"
                "    var dir = Input.get_vector(\"ui_left\",\"ui_right\",\"ui_up\",\"ui_down\")\n"
                "    velocity = dir * speed\n"
                "    if dir != Vector2.ZERO:\n"
                "        sprite.play(\"walk\")\n"
                "    else:\n"
                "        sprite.play(\"idle\")\n"
                "    move_and_slide()\n```"
            )
        return (
            "🎮 **AI Game Agent — Demo Mode**\n\n"
            "The server is running! To unlock full Claude AI power:\n"
            "1. Get a free key at https://console.anthropic.com\n"
            "2. Open `.env` in your project and paste:\n"
            "   `ANTHROPIC_API_KEY=sk-ant-...`\n"
            "3. Restart: `./start_agent.sh`\n\n"
            "Right now I can still:\n"
            "✅ Generate art via PixelLab MCP\n"
            "✅ Scaffold Godot project templates\n"
            "✅ Run Godot headless tests\n"
            "✅ Research isekai/anime ideas (no key needed)\n"
        )

    # ─── Ollama (local) ───────────────────────────────────────────────────────

    def _ollama(self, messages: list[dict]) -> str:
        import requests
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream": False,
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
