"""Central configuration for AI Game Agent."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Auto-load .env from project root
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
GODOT_PROJECT = PROJECT_ROOT / "godot_ai_colony"
GENERATED_GAMES = BASE_DIR / "generated_games"
TRAINING_DATA = BASE_DIR / "training_data"
TEMPLATES_DIR = BASE_DIR / "templates"

# Server
AGENT_HOST = "127.0.0.1"
AGENT_PORT = 8765

# AI mode: "api" uses Claude/GPT-4, "local" uses Ollama
AI_MODE = os.getenv("AI_MODE", "api")

# API keys — set in environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PIXELLAB_TOKEN = os.getenv("PIXELLAB_TOKEN", "355b7fcd-5355-4801-9dd5-df374f53501b")

# Local model (Ollama)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gdscript-assistant:latest")

# Godot CLI path — auto-detected for macOS
_GODOT_MACOS = "/Applications/Godot.app/Contents/MacOS/Godot"
import os as _os
GODOT_BIN = _os.getenv("GODOT_BIN", _GODOT_MACOS if Path(_GODOT_MACOS).exists() else "godot")

# Preferred LLM provider: "anthropic" | "openai" | "gemini" | "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
