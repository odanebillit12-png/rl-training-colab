"""
Self-improvement loop — generates a game, tests it headlessly,
auto-fixes errors with LLM, logs pairs as training data.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from datetime import datetime
from ai_game_agent.config import TRAINING_DATA
from ai_game_agent.orchestrator import LLMOrchestrator
from ai_game_agent.tools.godot_tools import scaffold_project, inject_script
from ai_game_agent.tools.godot_runner import run_headless


def _save_training_pair(prompt: str, error: str, fix: str, success: bool):
    TRAINING_DATA.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "error": error,
        "fix": fix,
        "success": success,
    }
    log = TRAINING_DATA / "pairs.jsonl"
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def self_improve_loop(
    game_name: str,
    game_type: str = "rpg",
    max_iterations: int = 5,
    research_context: str = "",
) -> dict:
    """
    Generate → test → fix loop.
    Returns the final result dict with project path + iteration count.
    """
    llm = LLMOrchestrator()
    iteration = 0
    project_path = None
    last_errors = []

    prompt_base = f"""
Generate a complete, playable Godot 4 {game_type} game called "{game_name}".

Requirements:
- Godot 4 GDScript only
- Top-down 2D view
- Player movement (WASD/arrow keys)
- At least one NPC or enemy
- Basic collision detection
- The main scene must be scenes/main.tscn

{f'Inspiration context:{chr(10)}{research_context}' if research_context else ''}

Return ONLY the GDScript code for scripts/player.gd and scripts/main.gd.
Format each as a separate code block with the filename as a comment on the first line:
```gdscript
# scripts/player.gd
...code...
```
"""

    while iteration < max_iterations:
        iteration += 1
        print(f"\n[SelfImprove] Iteration {iteration}/{max_iterations}")

        if iteration == 1:
            prompt = prompt_base
        else:
            error_text = "\n".join(last_errors[:10])
            prompt = f"""
The previous version had these Godot errors:
{error_text}

Fix ALL errors. Return the corrected GDScript files in the same format.
Keep the fix minimal — only change what causes the errors.
"""

        reply = llm.chat(prompt)
        blocks = llm.extract_code_blocks(reply)

        if not blocks:
            print("[SelfImprove] No code blocks in reply, retrying...")
            continue

        # Build or update project
        if project_path is None:
            project_path = scaffold_project(game_name, game_type)

        for block in blocks:
            code = block["code"]
            # Extract filename from first comment line
            lines = code.strip().splitlines()
            if lines and lines[0].startswith("# scripts/"):
                script_path = lines[0].lstrip("# ").strip()
                code = "\n".join(lines[1:])
                inject_script(project_path, script_path, code)
                print(f"  [Write] {script_path}")

        # Test
        print("[SelfImprove] Running headless test...")
        result = run_headless(project_path, timeout=20)
        print(f"  Success={result['success']}, Errors={len(result['errors'])}")

        # Log training pair
        error_summary = "; ".join(result["errors"][:5])
        _save_training_pair(
            prompt=prompt[:500],
            error=error_summary,
            fix=reply[:1000],
            success=result["success"],
        )

        if result["success"]:
            print(f"[SelfImprove] ✅ Game works after {iteration} iteration(s)!")
            return {
                "success": True,
                "project_path": str(project_path),
                "iterations": iteration,
                "warnings": result["warnings"],
            }

        last_errors = result["errors"]

    print("[SelfImprove] ❌ Max iterations reached.")
    return {
        "success": False,
        "project_path": str(project_path) if project_path else None,
        "iterations": iteration,
        "last_errors": last_errors,
    }
