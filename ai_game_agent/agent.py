"""
AI Game Agent Server — FastAPI endpoints for the Godot plugin + CLI.
Run with: python -m ai_game_agent.agent
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Optional

# Load .env from project root before anything else
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ai_game_agent.config import AGENT_HOST, AGENT_PORT, GENERATED_GAMES
from ai_game_agent.orchestrator import LLMOrchestrator
from ai_game_agent.tools.godot_tools import (
    scaffold_project, inject_script, list_project_scripts
)
from ai_game_agent.tools.godot_runner import run_headless
from ai_game_agent.tools.research_tools import (
    research_isekai_ideas, research_pixel_art_style,
    research_game_mechanics, summarise_research_for_llm
)
from ai_game_agent.self_improve import self_improve_loop
from ai_game_agent.tools.pixel_artist import (
    draw_character, draw_tile, draw_prop,
    draw_to_base64, draw_character_all_directions, PALETTES,
)
from ai_game_agent.tools.animator import (
    generate_character_sheet, SpriteAnimator, ANIMATIONS, frames_to_base64_list,
)

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Game Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLMOrchestrator()

# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = ""
    reset_history: bool = False

class ChatResponse(BaseModel):
    reply: str
    code_blocks: list[dict]

class AutoBuildRequest(BaseModel):
    game_name: str
    game_type: str = "rpg"           # rpg | platformer | sandbox
    description: str = ""
    research_topic: str = ""         # e.g. "isekai RPG village simulation"
    use_research: bool = True
    max_fix_iterations: int = 3
    generate_art: bool = False       # calls PixelLab if True

class BuildResponse(BaseModel):
    success: bool
    project_path: str
    iterations: int
    message: str
    errors: list[str] = []

class ResearchRequest(BaseModel):
    topic: str
    mode: str = "isekai"             # isekai | pixel_art | mechanic

class InjectScriptRequest(BaseModel):
    project_path: str
    script_path: str
    code: str

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    from ai_game_agent.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, LLM_PROVIDER, GODOT_BIN
    import os
    has_key = bool(ANTHROPIC_API_KEY or OPENAI_API_KEY or GEMINI_API_KEY)
    return {
        "status": "ok",
        "version": "0.1.0",
        "mode": "api" if has_key else "demo",
        "provider": LLM_PROVIDER,
        "godot": GODOT_BIN,
        "api_key_set": has_key,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Interactive chat — asks the AI anything about game dev."""
    if req.reset_history:
        llm.reset()
    try:
        reply = llm.chat(req.message, context=req.context or "")
        blocks = llm.extract_code_blocks(reply)
        return ChatResponse(reply=reply, code_blocks=blocks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build", response_model=BuildResponse)
def auto_build(req: AutoBuildRequest):
    """
    Autonomous game builder:
      1. Research isekai/anime/game inspiration (optional)
      2. Generate full Godot project via LLM
      3. Self-improve: test headlessly, auto-fix errors
    """
    context = ""

    if req.use_research and req.research_topic:
        try:
            topic = req.research_topic
            print(f"[Agent] Researching: {topic}")
            research = research_isekai_ideas(topic)
            context = summarise_research_for_llm(research)
            print("[Agent] Research complete.")
        except Exception as e:
            print(f"[Agent] Research failed (non-fatal): {e}")

    if req.description:
        context += f"\n\nGame description: {req.description}"

    result = self_improve_loop(
        game_name=req.game_name,
        game_type=req.game_type,
        max_iterations=req.max_fix_iterations,
        research_context=context,
    )

    msg = "Game built successfully!" if result["success"] else f"Built with {result['iterations']} iterations but errors remain."
    return BuildResponse(
        success=result["success"],
        project_path=result.get("project_path", ""),
        iterations=result["iterations"],
        message=msg,
        errors=result.get("last_errors", []),
    )


@app.post("/research")
def research(req: ResearchRequest):
    """Research anime, pixel art, or game mechanics for inspiration."""
    try:
        if req.mode == "isekai":
            raw = research_isekai_ideas(req.topic)
        elif req.mode == "pixel_art":
            raw = research_pixel_art_style(req.topic)
        else:
            raw = research_game_mechanics(req.topic)
        summary = summarise_research_for_llm(raw)
        return {"summary": summary, "raw": raw}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inject")
def inject(req: InjectScriptRequest):
    """Inject a GDScript file into an existing project."""
    try:
        result = inject_script(Path(req.project_path), req.script_path, req.code)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects")
def list_projects():
    """List all generated game projects."""
    if not GENERATED_GAMES.exists():
        return {"projects": []}
    projects = []
    for d in GENERATED_GAMES.iterdir():
        if d.is_dir() and (d / "project.godot").exists():
            scripts = list_project_scripts(d)
            projects.append({"name": d.name, "path": str(d), "scripts": scripts})
    return {"projects": projects}


@app.post("/test/{project_name}")
def test_project(project_name: str):
    """Run a generated project headlessly and return test results."""
    project_path = GENERATED_GAMES / project_name
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    result = run_headless(project_path)
    return result


@app.post("/generate-art")
def generate_art(description: str, art_type: str = "character"):
    """Queue PixelLab art generation."""
    from ai_game_agent.tools.pixellab_tools import generate_character, generate_tileset
    try:
        if art_type == "character":
            result = generate_character(description)
        else:
            result = generate_tileset(description, description + " elevated")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Pixel Art Drawing Endpoints ──────────────────────────────────────────────

class DrawCharacterRequest(BaseModel):
    archetype: str = "warrior"       # warrior, mage, archer, villager, goblin, elf, skeleton
    size: int = 32                   # 16 or 32
    direction: str = "south"         # south, north, east, west
    palette: str = "db32"            # db32, pico8, rpg
    all_directions: bool = False
    seed: int = 0

class DrawTileRequest(BaseModel):
    tile_type: str = "grass"         # grass, dirt, stone, water, sand, forest_floor, lava, snow, swamp, wood, brick
    size: int = 16
    seed: int = 0

class DrawPropRequest(BaseModel):
    prop_type: str = "tree"          # tree, chest, torch, barrel, sign, flower
    size: int = 32
    seed: int = 0

class AnimateRequest(BaseModel):
    archetype: str = "warrior"
    animation: str = "walk"          # idle, walk, run, attack, cast, death, jump, hurt, cheer
    direction: str = "south"
    size: int = 32
    palette: str = "db32"
    seed: int = 0

class SpriteSheetRequest(BaseModel):
    archetype: str = "warrior"
    animations: list[str] = ["idle", "walk", "run", "attack", "death"]
    all_directions: bool = True
    size: int = 32
    palette: str = "db32"
    seed: int = 0
    save: bool = True                # save PNG to outputs/sprites/


@app.post("/draw/character")
def api_draw_character(req: DrawCharacterRequest):
    """Draw a pixel art character — returns base64 PNG(s)."""
    try:
        if req.all_directions:
            imgs = draw_character_all_directions(req.archetype, req.size, req.palette, req.seed)
            return {
                "type": "character_all_directions",
                "archetype": req.archetype,
                "images": {d: draw_to_base64(img) for d, img in imgs.items()},
            }
        else:
            img = draw_character(req.archetype, req.size, req.direction, req.palette, req.seed)
            return {
                "type": "character",
                "archetype": req.archetype,
                "direction": req.direction,
                "image": draw_to_base64(img),
                "size": req.size,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/draw/tile")
def api_draw_tile(req: DrawTileRequest):
    """Draw a pixel art tile — returns base64 PNG."""
    try:
        img = draw_tile(req.tile_type, req.size, req.seed)
        return {"type": "tile", "tile_type": req.tile_type, "image": draw_to_base64(img), "size": req.size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/draw/prop")
def api_draw_prop(req: DrawPropRequest):
    """Draw a pixel art prop/object — returns base64 PNG."""
    try:
        img = draw_prop(req.prop_type, req.size, req.seed)
        return {"type": "prop", "prop_type": req.prop_type, "image": draw_to_base64(img), "size": req.size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/animate")
def api_animate(req: AnimateRequest):
    """Generate animation frames for a character — returns list of base64 PNGs."""
    try:
        animator = SpriteAnimator(req.archetype, req.size, req.palette, req.seed)
        frames = animator.animate(req.animation, req.direction)
        return {
            "type": "animation",
            "archetype": req.archetype,
            "animation": req.animation,
            "direction": req.direction,
            "frame_count": len(frames),
            "fps": ANIMATIONS.get(req.animation, {}).get("fps", 8),
            "frames": frames_to_base64_list(frames),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sprite-sheet")
def api_sprite_sheet(req: SpriteSheetRequest):
    """Generate a full sprite sheet with all animations — saves PNG + JSON."""
    try:
        out_dir = None
        if req.save:
            out_dir = str(Path(__file__).parent.parent / "outputs" / "sprites" / req.archetype)
        result = generate_character_sheet(
            archetype=req.archetype,
            size=req.size,
            animations=req.animations,
            all_directions=req.all_directions,
            output_dir=out_dir,
            palette=req.palette,
            seed=req.seed,
        )
        # Return preview of south sheet + metadata (not full pixel data for all directions)
        south_sheet = result["sheets"].get("south", list(result["sheets"].values())[0])
        return {
            "type": "sprite_sheet",
            "archetype": req.archetype,
            "saved_to": out_dir,
            "directions": list(result["sheets"].keys()),
            "animations": req.animations,
            "godot_meta": result["godot_meta"],
            "preview": draw_to_base64(south_sheet["sheet"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/draw/options")
def draw_options():
    """List all available archetypes, tiles, props, animations."""
    return {
        "archetypes": ["warrior", "mage", "archer", "villager", "goblin", "elf", "skeleton"],
        "tile_types": ["grass", "dirt", "stone", "water", "sand", "forest_floor", "lava", "snow", "swamp", "wood", "brick"],
        "props": ["tree", "chest", "torch", "barrel", "sign", "flower"],
        "animations": list(ANIMATIONS.keys()),
        "palettes": list(PALETTES.keys()),
        "directions": ["south", "north", "east", "west"],
    }


# ── Training / RL endpoints ───────────────────────────────────────────────────

from ai_game_agent.training.game_evaluator import GameEvaluator
from ai_game_agent.training.experience_memory import ExperienceMemory
from ai_game_agent.training.rl_trainer import RLTrainer, TrainingConfig, run_training_session
from ai_game_agent.training.pixellab_trainer import PixelLabConfig, PixelLabTrainer
import asyncio, threading, os

_training_state = {
    "running": False,
    "episode": 0,
    "score": 0.0,
    "rolling_avg": 0.0,
    "level": "Basics",
    "log": [],
    "stats": {},
}
_trainer_instance: RLTrainer | None = None
_memory = ExperienceMemory()
_evaluator = GameEvaluator()

# PixelLab trainer state (separate from RL trainer)
_pixellab_state = {
    "running": False,
    "episode": 0,
    "score": 0.0,
    "rolling_avg": 0.0,
    "level": "First Pixels",
    "log": [],
    "stats": {},
}
_pixellab_trainer: PixelLabTrainer | None = None


class TrainRequest(BaseModel):
    episodes: int = 50
    target_score: float = 75.0


class EvalRequest(BaseModel):
    image: Optional[str] = None      # base64 PNG
    code: Optional[str] = None
    description: Optional[str] = None


@app.post("/train/start")
def start_training(req: TrainRequest):
    """Start a training run in the background."""
    global _trainer_instance
    if _training_state["running"]:
        return {"status": "already_running", "episode": _training_state["episode"]}

    _training_state.update({"running": True, "episode": 0, "log": []})

    def on_episode(info: dict):
        _training_state.update({
            "episode": info["episode"],
            "score": info["score"],
            "rolling_avg": info["rolling_avg"],
            "level": info["level_name"],
        })
        entry = f"[Ep {info['episode']}] {info['status']} Score:{info['score']:.0f} Avg:{info['rolling_avg']:.0f} — {info['action']}"
        _training_state["log"] = ([entry] + _training_state["log"])[:50]

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cfg = TrainingConfig(max_episodes=req.episodes, target_score=req.target_score)
        trainer = RLTrainer(config=cfg, on_episode=on_episode)
        _trainer_instance = trainer
        stats = loop.run_until_complete(trainer.run())
        _training_state["running"] = False
        _training_state["stats"] = stats
        loop.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"status": "started", "episodes": req.episodes, "target_score": req.target_score}


@app.post("/train/stop")
def stop_training():
    """Stop an in-progress training run."""
    global _trainer_instance
    if _trainer_instance:
        _trainer_instance.stop()
    _training_state["running"] = False
    return {"status": "stopped"}


@app.get("/train/status")
def training_status():
    """Get current training progress."""
    stats = _memory.stats()
    return {
        **_training_state,
        "memory_stats": stats,
        "reward_curve": _memory.reward_curve()[-50:],  # last 50 scores
    }


@app.post("/train/evaluate")
def evaluate_output(req: EvalRequest):
    """Manually evaluate any output (image/code/description) and get a score."""
    result = _evaluator.evaluate(
        image=req.image,
        code=req.code or "",
        description=req.description or "",
    )
    return {
        "total_score": round(result.total_score, 1),
        "pixel_art_score": round(result.pixel_art_score, 1),
        "code_score": round(result.code_score, 1),
        "design_score": round(result.design_score, 1),
        "penalties": result.penalties,
        "bonuses": result.bonuses,
        "summary": result.summary(),
    }


@app.get("/train/memory")
def get_memory():
    """Return training memory stats + top and worst episodes."""
    top = _memory.top_examples(5)
    bad = _memory.bad_examples(5)
    return {
        "stats": _memory.stats(),
        "top_episodes": [
            {"score": e.total_score, "action": e.action_type, "lesson": e.lesson, "bonuses": e.bonuses[:3]}
            for e in top
        ],
        "bad_episodes": [
            {"score": e.total_score, "action": e.action_type, "lesson": e.lesson, "penalties": e.penalties[:3]}
            for e in bad
        ],
        "reward_curve": _memory.reward_curve(),
    }


# ── PixelLab Curriculum Training ───────────────────────────────────────────────

class PixelLabTrainRequest(BaseModel):
    episodes: int = 50
    target_score: float = 80.0
    api_key: Optional[str] = None   # falls back to PIXELLAB_API_KEY env var


@app.post("/train/pixellab/start")
def start_pixellab_training(req: PixelLabTrainRequest):
    """Start AI Dev PixelLab curriculum training in the background."""
    global _pixellab_trainer
    if _pixellab_state["running"]:
        return {"status": "already_running", "episode": _pixellab_state["episode"]}

    api_key = req.api_key or os.environ.get("PIXELLAB_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "PixelLab API key required. Pass api_key or set PIXELLAB_API_KEY env var."}

    _pixellab_state.update({"running": True, "episode": 0, "log": []})

    def on_episode(info: dict):
        _pixellab_state.update({
            "episode": info["episode"],
            "score": info["score"],
            "rolling_avg": info["rolling_avg"],
            "level": info["level_name"],
        })
        icon = "✅" if info["reward"] else ("❌" if info["penalty"] else "⚠️")
        entry = f"{icon} Ep{info['episode']} L{info['level']} {info['score']:.0f}/100 — {info['task_name']}"
        _pixellab_state["log"] = ([entry] + _pixellab_state["log"])[:50]

    def run():
        global _pixellab_trainer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cfg = PixelLabConfig(
                api_key=api_key,
                max_episodes=req.episodes,
                target_score=req.target_score,
            )
            _pixellab_trainer = PixelLabTrainer(cfg, on_episode=on_episode)
            stats = loop.run_until_complete(_pixellab_trainer.run())
            _pixellab_state["stats"] = stats
        except Exception as e:
            _pixellab_state["stats"] = {"error": str(e)}
        finally:
            _pixellab_state["running"] = False
            loop.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return {"status": "started", "episodes": req.episodes, "target_score": req.target_score}


@app.post("/train/pixellab/stop")
def stop_pixellab_training():
    """Stop PixelLab training."""
    if _pixellab_trainer:
        _pixellab_trainer.stop()
    _pixellab_state["running"] = False
    return {"status": "stopped"}


@app.get("/train/pixellab/status")
def pixellab_training_status():
    """Get current PixelLab training progress."""
    return {**_pixellab_state}


@app.get("/train/pixellab/assets")
def list_pixellab_assets():
    """List all images generated during PixelLab training."""
    from pathlib import Path
    art_dir = Path("training_data/pixellab_art")
    if not art_dir.exists():
        return {"assets": [], "total": 0}
    files = sorted(art_dir.glob("*.png"))
    return {
        "assets": [
            {"name": f.stem, "path": str(f), "size_bytes": f.stat().st_size}
            for f in files
        ],
        "total": len(files),
        "dir": str(art_dir),
    }


# ── Training Dashboard endpoints (used by Godot Training tab) ─────────────────

class TrainingStartRequest(BaseModel):
    duration_minutes: int = 30

class SelfImproveRequest(BaseModel):
    max_iterations: int = 5

@app.get("/training/status")
def training_dashboard_status():
    """Aggregate training state for the Godot Training tab."""
    state_file = Path("training_data/cloud_state.json")
    rl_file    = Path("training_data/rl_session.json")
    si_file    = Path("training_data/self_improve_summary.json")

    art_score   = 0.0
    world_score = 0.0
    anim_score  = 0.0
    rl_best     = 0.0
    phase       = "Phase 1: Art"
    recent_rewards: list = []
    last_episode: dict = {}
    npc_win_rates: dict = {}

    if state_file.exists():
        try:
            cs = json.loads(state_file.read_text())
            art_score   = float(cs.get("best_score", 0.0))
            world_score = float(cs.get("world_score", 0.0))
            anim_score  = float(cs.get("anim_score",  0.0))
            phase       = cs.get("phase", phase)
            recent_rewards = [float(v) for v in cs.get("recent_rewards", [])][-10:]
            last_episode   = cs.get("last_episode", {})
        except Exception:
            pass

    if rl_file.exists():
        try:
            rl = json.loads(rl_file.read_text())
            rl_best      = float(rl.get("draw_best_reward", 0.0))
            npc_win_rates = rl.get("npc_win_rates", {})
            if not recent_rewards:
                recent_rewards = [float(v) for v in rl.get("recent_rewards", [])][-10:]
            if not last_episode:
                last_episode = rl.get("last_episode", {})
        except Exception:
            pass

    si_data: dict = {}
    if si_file.exists():
        try:
            si_data = json.loads(si_file.read_text())
        except Exception:
            pass

    return {
        "art_score":      round(art_score, 1),
        "world_score":    round(world_score, 1),
        "anim_score":     round(anim_score, 1),
        "rl_draw_best":   round(rl_best, 2),
        "current_phase":  phase,
        "recent_rewards": recent_rewards,
        "last_episode":   last_episode,
        "npc_win_rates":  npc_win_rates,
        "self_improve":   si_data,
    }


@app.post("/training/start")
def training_start(req: TrainingStartRequest):
    """Trigger cloud_train.py in the background for the given duration."""
    import subprocess, threading
    def _run():
        subprocess.run(
            [sys.executable, "cloud_train.py",
             "--duration", str(req.duration_minutes * 60)],
            cwd=Path(__file__).parent.parent,
        )
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "duration_minutes": req.duration_minutes}


@app.post("/training/stop")
def training_stop():
    """Signal the training loop to stop (writes a stop flag file)."""
    flag = Path("training_data/stop_training.flag")
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("stop")
    return {"status": "stop_requested"}


@app.post("/self-improve")
def self_improve_endpoint(req: SelfImproveRequest):
    """Run the self-improvement loop against the Godot project."""
    import json
    from pathlib import Path
    from ai_game_agent.training.self_improve import SelfImproveTrainer

    godot_project = Path("godot_ai_colony")
    if not godot_project.exists():
        return {"success": False, "message": "godot_ai_colony not found"}

    trainer = SelfImproveTrainer(
        project_path=str(godot_project),
        max_iterations=req.max_iterations,
    )
    result = trainer.run()

    # persist summary for the status endpoint
    summary_path = Path("training_data/self_improve_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2))

    return result




# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"🎮 AI Game Agent starting on http://{AGENT_HOST}:{AGENT_PORT}")
    print("   Godot plugin can connect at: http://127.0.0.1:8765")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run("ai_game_agent.agent:app", host=AGENT_HOST, port=AGENT_PORT, reload=False)

if __name__ == "__main__":
    main()
