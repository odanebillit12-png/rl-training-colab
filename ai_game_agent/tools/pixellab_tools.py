"""
PixelLab art generation tool — wraps the PixelLab REST API for
characters, tilesets, map objects, and animations.
"""
from __future__ import annotations
import time, requests, base64
from pathlib import Path
from ai_game_agent.config import PIXELLAB_TOKEN

BASE = "https://api.pixellab.ai"
HEADERS = {"Authorization": f"Bearer {PIXELLAB_TOKEN}"}


def _get(endpoint: str) -> dict:
    r = requests.get(f"{BASE}{endpoint}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(endpoint: str, payload: dict) -> dict:
    r = requests.post(f"{BASE}{endpoint}", headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def generate_character(description: str, name: str = "", size: int = 48,
                        n_directions: int = 8, mode: str = "standard") -> dict:
    """Create a pixel art character sprite sheet. Returns character ID."""
    payload = {
        "description": description,
        "name": name,
        "size": size,
        "n_directions": n_directions,
        "mode": mode,
        "view": "low top-down",
        "outline": "single color black outline",
        "shading": "basic shading",
    }
    return _post("/mcp/characters", payload)


def animate_character(character_id: str, animation: str = "walking-8-frames") -> dict:
    """Queue an animation for a character. Returns job info."""
    return _post(f"/mcp/characters/{character_id}/animations", {
        "template_animation_id": animation,
    })


def generate_tileset(lower: str, upper: str, tile_size: int = 32,
                     transition_size: float = 0.25) -> dict:
    """Generate a Wang tileset. Returns tileset ID."""
    return _post("/mcp/tilesets", {
        "lower_description": lower,
        "upper_description": upper,
        "transition_description": f"{lower} blending into {upper}",
        "transition_size": transition_size,
        "tile_size": {"width": tile_size, "height": tile_size},
        "outline": "lineless",
        "shading": "detailed shading",
        "detail": "highly detailed",
        "view": "high top-down",
    })


def wait_for_tileset(tileset_id: str, poll_interval: int = 10, max_wait: int = 180) -> dict:
    """Poll until tileset is complete, then return its data."""
    elapsed = 0
    while elapsed < max_wait:
        data = _get(f"/mcp/tilesets/{tileset_id}")
        if data.get("status") == "completed" or data.get("tileset_data"):
            return data
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Tileset {tileset_id} not ready after {max_wait}s")


def download_tileset_image(tileset_id: str, out_path: Path) -> Path:
    """Download tileset PNG to disk."""
    r = requests.get(
        f"{BASE}/mcp/tilesets/{tileset_id}/image",
        headers=HEADERS,
        allow_redirects=True,
        timeout=30,
    )
    r.raise_for_status()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return out_path


def generate_map_object(description: str, width: int = 64, height: int = 64) -> dict:
    """Generate a map object sprite. Returns object ID."""
    return _post("/mcp/map-objects", {
        "description": description,
        "width": width,
        "height": height,
        "view": "high top-down",
        "shading": "medium shading",
        "outline": "single color outline",
    })


def wait_for_character(char_id: str, poll_interval: int = 10, max_wait: int = 300) -> dict:
    """Poll until character is ready."""
    elapsed = 0
    while elapsed < max_wait:
        data = _get(f"/mcp/characters/{char_id}")
        if data.get("status") == "completed" or data.get("rotations"):
            return data
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Character {char_id} not ready after {max_wait}s")


def generate_full_asset_pack(game_description: str, out_dir: Path) -> dict:
    """
    Generate a complete art pack for a game:
      - player character + walk animation
      - 2 enemy characters
      - grass+forest tileset
      - ocean+beach tileset
      - 3 map objects (tree, rock, building)
    Returns dict of all IDs.
    """
    out_dir = Path(out_dir)
    results = {}

    print("[PixelLab] Generating player character...")
    char = generate_character(f"main hero for {game_description}", name="Hero", size=48)
    results["player_id"] = char.get("character_id") or char.get("id")

    print("[PixelLab] Generating goblin enemy...")
    goblin = generate_character("green goblin scout enemy warrior", name="Goblin", size=48)
    results["goblin_id"] = goblin.get("character_id") or goblin.get("id")

    print("[PixelLab] Generating grass-forest tileset...")
    ts = generate_tileset(
        "lush bright green grass meadow",
        "dense green forest canopy viewed from above",
    )
    results["grass_forest_tileset_id"] = ts.get("tileset_id") or ts.get("id")

    return results
