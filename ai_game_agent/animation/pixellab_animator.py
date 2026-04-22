"""
PixelLab Animator — AI Dev's complete guide to ALL PixelLab tools.

AI Dev uses this to:
  1. Create characters (standard + pro mode)
  2. Generate template animations (1 gen/direction — cheap!)
  3. Generate custom animations (20-40 gens — expensive, ask first!)
  4. Create tiles (isometric, topdown, sidescroller, pro)
  5. Create map objects for world building
  6. Download and validate results

AI Dev knows ALL PixelLab tools and picks the right one for each job.
"""
from __future__ import annotations
import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


PIXELLAB_BASE_URL = "https://api.pixellab.ai/v1"

# ── Complete tool reference — AI Dev studies this ─────────────────────────────

TOOL_REFERENCE = {
    "create_character": {
        "purpose": "Generate 4 or 8 direction character sprites",
        "modes": {
            "standard": "1 gen/direction — fast, template-based skeleton",
            "pro": "20-40 gens — 8 directions, higher quality AI reference",
        },
        "body_types": ["humanoid", "quadruped"],
        "quadruped_templates": ["bear", "cat", "dog", "horse", "lion"],
        "views": ["low top-down", "high top-down", "side"],
        "sizes": "16-128px (canvas ~40% larger)",
        "when_to_use": "Creating any character, NPC, or enemy sprite from scratch",
    },
    "animate_character": {
        "purpose": "Add animations to an existing character",
        "modes": {
            "template": "1 gen/direction — use pre-built animation ID (walk, run, kick, etc.)",
            "custom": "20-40 gens/direction — AI generates from description (expensive!)",
        },
        "humanoid_templates": [
            "walk", "walking", "walking-4-frames", "walking-6-frames", "walking-8-frames",
            "running-4-frames", "running-6-frames", "running-8-frames",
            "jumping-1", "jumping-2", "running-jump",
            "fight-stance-idle-8-frames", "breathing-idle",
            "lead-jab", "cross-punch", "roundhouse-kick", "high-kick",
            "flying-kick", "hurricane-kick", "leg-sweep",
            "fireball", "throw-object", "taking-punch",
            "crouching", "crouched-walking",
            "front-flip", "backflip", "running-slide",
            "falling-back-death", "getting-up",
            "picking-up", "drinking", "pushing", "pull-heavy-object",
            "sad-walk", "scary-walk", "surprise-uppercut", "two-footed-jump",
        ],
        "when_to_use": (
            "ALWAYS prefer template animations (1 gen each). "
            "Only use custom when no template exists for the action. "
            "Custom requires explicit user approval due to high cost."
        ),
    },
    "create_isometric_tile": {
        "purpose": "Single isometric tile — block/thick/thin tile shapes",
        "sizes": "16-64px",
        "when_to_use": "Creating individual dungeon tiles, blocks, items for isometric maps",
    },
    "create_topdown_tileset": {
        "purpose": "Wang tileset — 16 or 23 tiles for seamless top-down terrain transitions",
        "tile_sizes": "16px or 32px",
        "when_to_use": "Creating terrain transitions: ocean→beach, grass→dirt, snow→rock etc.",
    },
    "create_sidescroller_tileset": {
        "purpose": "Platform tileset for 2D side-view games",
        "tile_sizes": "16px or 32px",
        "when_to_use": "Platformer game terrain — stone platforms, wood, metal with ground/surface layers",
    },
    "create_tiles_pro": {
        "purpose": "AI fills pre-drawn tile shape outlines — hex/isometric/square/octagon",
        "tile_types": ["hex", "hex_pointy", "isometric", "octagon", "square_topdown"],
        "when_to_use": "Creating multiple matching tile variations from a description in one generation",
    },
    "create_map_object": {
        "purpose": "Standalone object with transparent background — trees, barrels, furniture",
        "modes": {
            "basic": "No background — standalone object",
            "style_match": "Provide background image to match existing map art style",
        },
        "when_to_use": "Adding props, furniture, trees, NPCs to existing maps",
    },
}


class PixelLabAnimator:
    """AI Dev's interface to all PixelLab generation and animation tools."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("PIXELLAB_API_KEY", "")
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    # ── Character Creation ──────────────────────────────────────────────────

    def create_character(
        self,
        description: str,
        name: str = "",
        mode: str = "standard",
        body_type: str = "humanoid",
        n_directions: int = 8,
        size: int = 48,
        view: str = "low top-down",
        outline: str = "single color black outline",
        shading: str = "basic shading",
        detail: str = "medium detail",
        proportions: str = '{"type":"preset","name":"default"}',
        template: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a character and return the character_id.
        Pro mode costs 20-40 gens — use standard (1 gen) for training.
        """
        payload = {
            "description": description,
            "name": name or description[:30],
            "mode": mode,
            "body_type": body_type,
            "n_directions": n_directions,
            "size": size,
            "view": view,
            "outline": outline,
            "shading": shading,
            "detail": detail,
            "proportions": proportions,
        }
        if body_type == "quadruped" and template:
            payload["template"] = template

        resp = self._post("/characters", payload)
        if resp and "character_id" in resp:
            return resp["character_id"]
        return None

    def get_character(self, character_id: str, max_wait: int = 300) -> Optional[Dict]:
        """Poll until character is ready and return full data."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            resp = self._get(f"/characters/{character_id}")
            if not resp:
                break
            status = resp.get("status", "")
            if status == "completed":
                return resp
            elif status == "failed":
                print(f"    ❌ Character generation failed: {resp.get('error')}")
                return None
            time.sleep(10)
        print(f"    ⚠️  Character {character_id} timed out after {max_wait}s")
        return None

    # ── Template Animation (cheap — 1 gen/direction) ────────────────────────

    def animate_template(
        self,
        character_id: str,
        template_id: str,
        directions: Optional[List[str]] = None,
        animation_name: str = "",
    ) -> List[str]:
        """
        Queue template animation jobs. Returns list of job IDs.
        CHEAP: 1 generation per direction.
        """
        payload: Dict[str, Any] = {
            "character_id": character_id,
            "template_animation_id": template_id,
        }
        if directions:
            payload["directions"] = directions
        if animation_name:
            payload["animation_name"] = animation_name

        resp = self._post("/characters/animate", payload)
        if resp:
            return resp.get("job_ids", [])
        return []

    def animate_custom(
        self,
        character_id: str,
        action_description: str,
        directions: Optional[List[str]] = None,
        animation_name: str = "",
        confirmed: bool = False,
    ) -> Tuple[Optional[List[str]], int]:
        """
        Queue custom animation (expensive — 20-40 gens/direction).
        Returns (job_ids, estimated_cost). Only sends if confirmed=True.
        """
        n_dirs = len(directions) if directions else 1
        est_cost = n_dirs * 30  # ~30 gens per direction average

        if not confirmed:
            return None, est_cost

        payload: Dict[str, Any] = {
            "character_id": character_id,
            "action_description": action_description,
            "confirm_cost": True,
        }
        if directions:
            payload["directions"] = directions
        if animation_name:
            payload["animation_name"] = animation_name

        resp = self._post("/characters/animate", payload)
        if resp:
            return resp.get("job_ids", []), est_cost
        return None, est_cost

    def wait_for_animation(self, character_id: str, max_wait: int = 300) -> Optional[Dict]:
        """Poll character until latest animation is completed."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            char = self._get(f"/characters/{character_id}")
            if not char:
                break
            animations = char.get("animations", [])
            if animations:
                latest = animations[-1]
                if latest.get("status") == "completed":
                    return latest
                elif latest.get("status") == "failed":
                    print(f"    ❌ Animation failed: {latest.get('error')}")
                    return None
            pending = char.get("pending_jobs", [])
            if not pending and animations and animations[-1].get("status") != "completed":
                return None
            time.sleep(15)
        return None

    def download_animation_frames(
        self,
        animation_data: Dict,
        save_dir: Path,
        animation_name: str,
    ) -> List[Path]:
        """
        Download animation sprite sheet images from a completed animation.
        Returns list of saved PNG paths (one per direction).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        directions = animation_data.get("directions", {})
        for direction, dir_data in directions.items():
            url = dir_data.get("spritesheet_url") or dir_data.get("image_url")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    clean_dir = direction.replace("-", "_")
                    clean_name = animation_name.replace(" ", "_").lower()
                    path = save_dir / f"{clean_name}_{clean_dir}.png"
                    path.write_bytes(resp.content)
                    saved.append(path)
            except Exception as e:
                print(f"    ⚠️  Failed to download {direction}: {e}")

        return saved

    # ── Tileset & Tile Tools ─────────────────────────────────────────────────

    def create_topdown_tileset(
        self,
        lower_desc: str,
        upper_desc: str,
        transition_desc: str = "",
        tile_size: int = 32,
        transition_size: float = 0.25,
        shading: str = "basic shading",
        detail: str = "medium detail",
    ) -> Optional[str]:
        """Create Wang tileset for top-down terrain. Returns tileset_id."""
        payload = {
            "lower_description": lower_desc,
            "upper_description": upper_desc,
            "tile_size": {"width": tile_size, "height": tile_size},
            "transition_size": transition_size,
            "shading": shading,
            "detail": detail,
        }
        if transition_desc:
            payload["transition_description"] = transition_desc
        resp = self._post("/tilesets/topdown", payload)
        return resp.get("tileset_id") if resp else None

    def create_sidescroller_tileset(
        self,
        lower_desc: str,
        transition_desc: str,
        tile_size: int = 16,
        transition_size: float = 0.25,
    ) -> Optional[str]:
        """Create sidescroller platform tileset. Returns tileset_id."""
        payload = {
            "lower_description": lower_desc,
            "transition_description": transition_desc,
            "tile_size": {"width": tile_size, "height": tile_size},
            "transition_size": transition_size,
        }
        resp = self._post("/tilesets/sidescroller", payload)
        return resp.get("tileset_id") if resp else None

    def create_isometric_tile(
        self,
        description: str,
        size: int = 32,
        tile_shape: str = "block",
        shading: str = "basic shading",
    ) -> Optional[str]:
        """Create single isometric tile. Returns tile_id."""
        payload = {
            "description": description,
            "size": size,
            "tile_shape": tile_shape,
            "shading": shading,
        }
        resp = self._post("/tiles/isometric", payload)
        return resp.get("tile_id") if resp else None

    def create_map_object(
        self,
        description: str,
        width: int = 64,
        height: int = 64,
        view: str = "high top-down",
        shading: str = "medium shading",
    ) -> Optional[str]:
        """Create map prop/object with transparent background. Returns object_id."""
        payload = {
            "description": description,
            "width": width,
            "height": height,
            "view": view,
            "shading": shading,
        }
        resp = self._post("/map-objects", payload)
        return resp.get("object_id") if resp else None

    def wait_for_tileset(self, tileset_id: str, endpoint: str = "topdown",
                         max_wait: int = 180) -> Optional[Dict]:
        """Poll tileset until complete. endpoint = topdown|sidescroller."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            resp = self._get(f"/tilesets/{endpoint}/{tileset_id}")
            if not resp:
                break
            if resp.get("status") == "completed":
                return resp
            elif resp.get("status") == "failed":
                return None
            time.sleep(15)
        return None

    # ── API helpers ───────────────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: Dict) -> Optional[Dict]:
        if not self.api_key:
            return None
        try:
            resp = self._session.post(f"{PIXELLAB_BASE_URL}{endpoint}", json=payload, timeout=30)
            if resp.status_code in (200, 201, 202):
                return resp.json()
            print(f"    ⚠️  PixelLab POST {endpoint} → {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"    ⚠️  PixelLab POST error: {e}")
        return None

    def _get(self, endpoint: str) -> Optional[Dict]:
        if not self.api_key:
            return None
        try:
            resp = self._session.get(f"{PIXELLAB_BASE_URL}{endpoint}", timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"    ⚠️  PixelLab GET error: {e}")
        return None

    def is_available(self) -> bool:
        """Check if PixelLab API key is set and responsive."""
        if not self.api_key:
            return False
        resp = self._get("/balance")
        return resp is not None

    def get_balance(self) -> int:
        """Return remaining generation credits."""
        resp = self._get("/balance")
        if resp:
            return resp.get("balance", 0)
        return 0
