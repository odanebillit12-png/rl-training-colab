"""
Godot headless runner — launches Godot CLI to test generated projects,
captures stdout/stderr, returns structured results.
"""
from __future__ import annotations
import subprocess, time, os
from pathlib import Path
from ai_game_agent.config import GODOT_BIN


def find_godot() -> str:
    """Auto-detect Godot 4 binary on macOS / Linux / Windows."""
    candidates = [
        GODOT_BIN,
        "/Applications/Godot.app/Contents/MacOS/Godot",
        "/Applications/Godot_v4.app/Contents/MacOS/Godot",
        "/usr/local/bin/godot",
        "/usr/bin/godot",
        "godot4",
        "godot",
    ]
    for c in candidates:
        try:
            r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return GODOT_BIN  # fallback — user must set GODOT_BIN


GODOT = find_godot()


def run_headless(project_path: Path, timeout: int = 30) -> dict:
    """
    Run a Godot project headlessly for `timeout` seconds.
    Returns:
        {
          "success": bool,
          "errors": [str],
          "warnings": [str],
          "output": str,
          "crash": bool,
        }
    """
    project_path = Path(project_path)
    if not (project_path / "project.godot").exists():
        return {"success": False, "errors": ["project.godot not found"], "warnings": [], "output": "", "crash": False}

    cmd = [GODOT, "--headless", "--path", str(project_path), "--quit"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout + result.stderr
        errors   = [l for l in stdout.splitlines() if "ERROR"   in l or "SCRIPT ERROR" in l]
        warnings = [l for l in stdout.splitlines() if "WARNING" in l]
        success  = result.returncode == 0 and not errors
        return {
            "success": success,
            "errors": errors,
            "warnings": warnings,
            "output": stdout[:4000],
            "crash": result.returncode not in (0, 1),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "errors": ["Godot timed out"], "warnings": [], "output": "", "crash": False}
    except FileNotFoundError:
        return {"success": False, "errors": [f"Godot binary not found: {GODOT}"], "warnings": [], "output": "", "crash": False}


def check_script_syntax(script_path: Path) -> dict:
    """Use Godot to check a single GDScript file for syntax errors."""
    cmd = [GODOT, "--headless", "--check-only", "--script", str(script_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = r.stdout + r.stderr
        errors = [l for l in out.splitlines() if "ERROR" in l or "Parse Error" in l]
        return {"valid": len(errors) == 0, "errors": errors, "output": out}
    except Exception as e:
        return {"valid": False, "errors": [str(e)], "output": ""}
