# AI Game Making Assistant

An AI that **builds games from scratch** using Godot 4. Two modes:
- 🎮 **AutoBuild** — describe a game, it generates the full project automatically
- 💬 **Chat** — interactive assistant inside the Godot editor

Starts using Claude/GPT-4 API, then fine-tunes to a local offline model.

---

## Architecture

```
Godot Editor Plugin  ←HTTP→  Python Agent Server (localhost:8765)
                                  ├── LLM (Claude API / local Ollama)
                                  ├── Research (YouTube + AniList isekai + web)
                                  ├── GDScript generator + scene builder
                                  ├── PixelLab art generator
                                  └── Headless test loop (self-improvement)
```

---

## Quick Start

### 1. Install Python dependencies
```bash
cd /Users/odanebillit/Documents/ScikitPro/ai_game_agent
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
export LLM_PROVIDER=openai
```

### 3. Start the agent server
```bash
cd /Users/odanebillit/Documents/ScikitPro
python -m ai_game_agent.agent
# → Server running at http://127.0.0.1:8765
```

### 4. Enable the Godot plugin
1. Open `godot_ai_colony` in Godot 4
2. Go to **Project → Project Settings → Plugins**
3. Enable **AI Game Assistant**
4. The panel appears in the bottom-right dock

---

## Godot Plugin Tabs

| Tab | What it does |
|-----|-------------|
| **Chat** | Ask anything — get GDScript code, scene ideas, balance advice |
| **AutoBuild** | Name + describe your game → full project generated + tested automatically |
| **Research** | Search isekai anime, YouTube tutorials, game design articles for inspiration |

---

## AutoBuild flow

1. Enter game name + type (RPG / Platformer / Sandbox)
2. Optionally describe it ("isekai village with monster invasions")
3. Check "Use anime/game research" — agent searches AniList + YouTube first
4. Click **AUTO-BUILD GAME**
5. Agent generates → tests with Godot headless → auto-fixes errors → delivers project

Generated projects are saved to:
```
ai_game_agent/generated_games/<game_name>/
  project.godot
  scenes/main.tscn
  scripts/player.gd
  scripts/...
  assets/
```

---

## Fine-tune Local Model (Google Colab)

Open `colab/gdscript_finetune.ipynb` in Google Colab (free T4 GPU):

1. Run all cells — downloads GDScript from GitHub, fine-tunes CodeLlama-7B with LoRA
2. Export GGUF model
3. Install [Ollama](https://ollama.ai) locally
4. `ollama create gdscript-assistant -f Modelfile`
5. Switch agent to local mode:
```bash
export AI_MODE=local
export OLLAMA_MODEL=gdscript-assistant
python -m ai_game_agent.agent
```

---

## Self-Improvement Loop

The agent automatically improves its code:
```
Generate game → Run Godot headless → Errors? → LLM fixes → Re-test → Repeat
```
All (prompt, error, fix) pairs are saved to `ai_game_agent/training_data/pairs.jsonl`
and used in the next Colab training run.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `AI_MODE` | `api` | `api` (cloud) \| `local` (Ollama) |
| `OLLAMA_MODEL` | `gdscript-assistant` | Ollama model name |
| `GODOT_BIN` | `godot` | Path to Godot 4 binary |
| `PIXELLAB_TOKEN` | (set) | PixelLab API token |

---

## Roadmap

- [x] Python agent server (FastAPI)
- [x] Godot editor plugin (Chat + AutoBuild + Research)
- [x] Isekai/anime research (AniList + YouTube + web)
- [x] GDScript generator + scene scaffolder
- [x] PixelLab art integration
- [x] Self-improvement loop (generate → test → fix)
- [x] Google Colab fine-tuning notebook (CodeLlama-7B LoRA)
- [ ] Unity C# support
- [ ] RPG Maker JS plugin
- [ ] Voice input ("just say what game you want")
- [ ] Streaming responses in Godot plugin
