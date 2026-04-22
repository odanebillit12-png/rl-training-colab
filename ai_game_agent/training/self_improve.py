"""
self_improve.py — Autonomous self-improvement loop.

Flow per iteration:
  1. Run Godot headless → capture stdout/stderr
  2. Parse errors (GDScript, shader, scene parse, crash)
  3. Map errors to source files
  4. Send {file, error, context} to LLM → get patch
  5. Apply patch
  6. Repeat until clean run or max_iterations reached
  7. Save pairs (error, fix) as training data

Usage:
    python3 -m ai_game_agent.training.self_improve
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


@dataclass
class SelfImproveConfig:
    godot_path:     str   = "godot"          # path to godot binary
    project_path:   str   = "godot_ai_colony"
    max_iterations: int   = 10
    timeout_sec:    int   = 30               # headless run timeout
    training_dir:   str   = "training_data/self_improve"
    gemini_api_key: str   = ""
    groq_api_key:   str   = ""
    anthropic_key:  str   = ""


@dataclass
class GodotError:
    file:    str
    line:    int
    message: str
    kind:    str   # SCRIPT / SHADER / PARSE / CRASH


@dataclass
class FixResult:
    file:       str
    original:   str
    patched:    str
    error:      str
    success:    bool = False


class SelfImproveTrainer:
    """Autonomous game-fix loop."""

    def __init__(self, config: SelfImproveConfig):
        self.config = config
        self.on_iteration: Optional[Callable] = None
        Path(config.training_dir).mkdir(parents=True, exist_ok=True)
        self._pairs: List[dict] = []    # (error, fix) training data

    async def run(self) -> dict:
        stats = {
            "iterations": 0, "errors_fixed": 0,
            "errors_remaining": 0, "clean_run": False,
            "training_pairs": 0,
        }

        for i in range(self.config.max_iterations):
            stats["iterations"] += 1

            # 1. Run Godot headless
            output, crashed = await self._run_headless()

            # 2. Parse errors
            errors = self._parse_errors(output, crashed)

            if not errors:
                stats["clean_run"] = True
                stats["errors_remaining"] = 0
                if self.on_iteration:
                    self.on_iteration({"iteration": i+1, "errors": 0, "status": "CLEAN ✅"})
                break

            if self.on_iteration:
                self.on_iteration({
                    "iteration": i + 1,
                    "errors": len(errors),
                    "status": f"Fixing {len(errors)} error(s)...",
                    "first_error": f"{errors[0].file}:{errors[0].line} {errors[0].message[:60]}",
                })

            # 3. Fix each unique file mentioned in errors
            fixed = 0
            files_seen: set = set()
            for err in errors:
                if err.file in files_seen:
                    continue
                files_seen.add(err.file)

                file_errors = [e for e in errors if e.file == err.file]
                result = await self._fix_file(err.file, file_errors)
                if result.success:
                    fixed += 1
                    self._save_pair(result)

            stats["errors_fixed"] += fixed
            stats["errors_remaining"] = len(errors) - fixed

            await asyncio.sleep(0.5)

        stats["training_pairs"] = len(self._pairs)
        self._flush_training_data()
        return stats

    # ── Headless runner ─────────────────────────────────────────────────────

    async def _run_headless(self) -> tuple[str, bool]:
        """Run Godot headless and return (combined output, crashed)."""
        godot = self._find_godot()
        if not godot:
            return "ERROR: Godot binary not found", True

        cmd = [godot, "--headless", "--quit",
               "--path", str(Path(self.config.project_path).resolve())]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), self.config.timeout_sec)
                output = out.decode(errors="replace")
                crashed = proc.returncode not in (0, 256)  # 256 = normal headless exit
                return output, crashed
            except asyncio.TimeoutError:
                proc.kill()
                return "TIMEOUT: Godot did not exit in time", True
        except FileNotFoundError:
            return f"ERROR: Could not launch {godot}", True

    def _find_godot(self) -> Optional[str]:
        """Find Godot binary on PATH or common locations."""
        candidates = [
            self.config.godot_path,
            "godot4", "godot4-stable",
            "/usr/local/bin/godot",
            "/Applications/Godot.app/Contents/MacOS/Godot",
            "/Applications/Godot_v4.app/Contents/MacOS/Godot",
        ]
        for c in candidates:
            try:
                r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
                if r.returncode == 0:
                    return c
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    # ── Error parser ────────────────────────────────────────────────────────

    _PATTERNS = [
        # GDScript runtime: ERROR: res://scripts/player.gd:42
        (re.compile(r"ERROR:\s*(.+?\.gd):(\d+).*?:\s*(.+)", re.IGNORECASE), "SCRIPT"),
        # Parse error: Parse Error: "res://scenes/main.tscn"
        (re.compile(r"Parse Error.*?\"?(res://[^\":\s]+)\"?:(\d+).*?:\s*(.+)", re.IGNORECASE), "PARSE"),
        # Shader: shader_cache or material error
        (re.compile(r"SHADER.*?(res://[^:\s]+\.(?:gdshader|shader)):(\d+).*?:\s*(.+)", re.IGNORECASE), "SHADER"),
        # Generic res:// file error
        (re.compile(r"(res://[^\s:]+\.gd):(\d+).*?-\s*(.+)", re.IGNORECASE), "SCRIPT"),
    ]

    def _parse_errors(self, output: str, crashed: bool) -> List[GodotError]:
        errors: List[GodotError] = []
        seen = set()

        for pattern, kind in self._PATTERNS:
            for m in pattern.finditer(output):
                file, line_str, msg = m.group(1), m.group(2), m.group(3)
                key = (file, line_str)
                if key in seen:
                    continue
                seen.add(key)
                errors.append(GodotError(
                    file=file, line=int(line_str), message=msg.strip(), kind=kind
                ))

        if crashed and not errors:
            errors.append(GodotError(
                file="unknown", line=0,
                message=output[-500:] if output else "crash with no output",
                kind="CRASH"
            ))

        return errors

    # ── LLM-powered fixer ───────────────────────────────────────────────────

    async def _fix_file(self, res_path: str, errors: List[GodotError]) -> FixResult:
        """Read file, ask LLM for fix, write patched version."""
        local = self._res_to_local(res_path)
        if not local or not local.exists():
            return FixResult(file=res_path, original="", patched="", error="file not found")

        original = local.read_text(encoding="utf-8")

        error_descriptions = "\n".join(
            f"  Line {e.line} [{e.kind}]: {e.message}" for e in errors
        )

        prompt = f"""You are fixing a Godot 4 GDScript file that has runtime errors.

FILE: {res_path}
ERRORS:
{error_descriptions}

CURRENT CODE:
```gdscript
{original[:3000]}
```

Return ONLY the complete fixed GDScript file with no explanation, no markdown fences, no commentary.
Fix only the errors listed. Do not change unrelated code."""

        patched = await self._call_llm(prompt)
        if not patched or len(patched) < 20:
            return FixResult(file=res_path, original=original, patched="", error="LLM returned empty")

        # Strip accidental markdown fences
        patched = re.sub(r"^```[a-z]*\n?", "", patched, flags=re.MULTILINE)
        patched = re.sub(r"\n?```$", "", patched, flags=re.MULTILINE)
        patched = patched.strip()

        local.write_text(patched, encoding="utf-8")
        return FixResult(file=res_path, original=original, patched=patched,
                         error="", success=True)

    async def _call_llm(self, prompt: str) -> str:
        """Try Groq → Gemini → Anthropic in order."""
        if self.config.groq_api_key:
            result = await self._groq(prompt)
            if result:
                return result
        if self.config.gemini_api_key:
            result = await self._gemini(prompt)
            if result:
                return result
        if self.config.anthropic_key:
            result = await self._anthropic(prompt)
            if result:
                return result
        return ""

    async def _groq(self, prompt: str) -> str:
        try:
            import groq
            client = groq.AsyncGroq(api_key=self.config.groq_api_key)
            resp = await client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    async def _gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = await asyncio.to_thread(model.generate_content, prompt)
            return resp.text or ""
        except Exception:
            return ""

    async def _anthropic(self, prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=self.config.anthropic_key)
            resp = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text if resp.content else ""
        except Exception:
            return ""

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _res_to_local(self, res_path: str) -> Optional[Path]:
        """Convert res://scripts/foo.gd → godot_ai_colony/scripts/foo.gd"""
        rel = res_path.replace("res://", "")
        return Path(self.config.project_path) / rel

    def _save_pair(self, result: FixResult):
        self._pairs.append({
            "file": result.file,
            "original": result.original[:2000],
            "patched": result.patched[:2000],
        })

    def _flush_training_data(self):
        if not self._pairs:
            return
        out = Path(self.config.training_dir) / "fix_pairs.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            for pair in self._pairs:
                f.write(json.dumps(pair) + "\n")
        self._pairs.clear()


if __name__ == "__main__":
    async def _main():
        cfg = SelfImproveConfig(
            gemini_api_key  = os.environ.get("GEMINI_API_KEY",    ""),
            groq_api_key    = os.environ.get("GROQ_API_KEY",      ""),
            anthropic_key   = os.environ.get("ANTHROPIC_API_KEY", ""),
            max_iterations  = 5,
        )

        def on_iter(info):
            print(f"  Iteration {info['iteration']}: {info['status']}")
            if info.get('first_error'):
                print(f"    → {info['first_error']}")

        trainer = SelfImproveTrainer(cfg)
        trainer.on_iteration = on_iter
        stats = await trainer.run()
        print(f"\n{'='*50}")
        print(f"Iterations : {stats['iterations']}")
        print(f"Errors fixed: {stats['errors_fixed']}")
        print(f"Remaining  : {stats['errors_remaining']}")
        print(f"Clean run  : {'✅' if stats['clean_run'] else '❌'}")
        print(f"Train pairs: {stats['training_pairs']}")

    asyncio.run(_main())
