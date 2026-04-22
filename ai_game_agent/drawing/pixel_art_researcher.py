"""
PixelArtResearcher — gives AI Dev visual reference context before drawing.

Sources (no API key needed):
  1. OpenGameArt.org — search for similar sprites, pull style descriptions
  2. Lospec.com      — pixel art palettes matching the task subject
  3. Fetch fallback  — scrape any URL for text context

Optional (needs free Brave Search API key):
  BRAVE_API_KEY in env → richer web search for tutorials/references
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from typing import Optional

# Free tier: 2,000 queries/month — get key at https://api.search.brave.com/
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 8


class PixelArtResearcher:
    """
    Before AI Dev draws, call research(task) to get a reference context string.
    This context is injected into the drawing prompt so the LLM has style guidance.
    """

    def __init__(self, brave_key: str = ""):
        self.brave_key = brave_key or BRAVE_API_KEY
        self._cache: dict = {}

    # ──────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────

    def research(self, task: dict) -> str:
        """
        Returns a short reference context string (≤400 chars) for the drawing prompt.
        task = {"subject": str, "width": int, "height": int, ...}
        """
        subject: str = task.get("subject", "")
        name: str = task.get("name", subject)
        cache_key = name.lower().replace(" ", "_")

        if cache_key in self._cache:
            return self._cache[cache_key]

        context_parts = []

        # 1. Try Brave Search for pixel art tutorials/style tips
        brave_result = self._brave_search(f"pixel art {name} sprite tutorial style guide")
        if brave_result:
            context_parts.append(f"Web reference: {brave_result}")

        # 2. OpenGameArt — search for similar assets
        oga_result = self._search_opengameart(name)
        if oga_result:
            context_parts.append(f"OpenGameArt: {oga_result}")

        # 3. Lospec palette suggestion
        lospec_result = self._search_lospec(name)
        if lospec_result:
            context_parts.append(f"Palette tip: {lospec_result}")

        # 4. Static tips as fallback (always included)
        context_parts.append(self._static_tips(name, task))

        context = " | ".join(filter(None, context_parts))[:500]
        self._cache[cache_key] = context
        return context

    # ──────────────────────────────────────────────────────────────
    # BRAVE SEARCH
    # ──────────────────────────────────────────────────────────────

    def _brave_search(self, query: str) -> str:
        if not self.brave_key:
            return ""
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    **HEADERS,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_key,
                },
                params={"q": query, "count": 3, "text_decorations": False},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                snippets = []
                for result in data.get("web", {}).get("results", [])[:2]:
                    desc = result.get("description", "").strip()
                    if desc:
                        snippets.append(desc[:120])
                return " ".join(snippets)[:240]
        except Exception:
            pass
        return ""

    # ──────────────────────────────────────────────────────────────
    # OPENGAMEART.ORG
    # ──────────────────────────────────────────────────────────────

    def _search_opengameart(self, name: str) -> str:
        try:
            query = name.replace(" ", "+")
            url = f"https://opengameart.org/art-search-advanced?keys={query}&field_art_type_tid[]=9"
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                return ""
            soup = BeautifulSoup(resp.text, "html.parser")
            # Grab first result title + description
            results = soup.select(".field-type-text-with-summary")
            if results:
                text = results[0].get_text(" ", strip=True)[:200]
                return re.sub(r"\s+", " ", text)
            # Try titles
            titles = soup.select("h3.field-content a")
            if titles:
                return f"Found: {titles[0].get_text(strip=True)}"
        except Exception:
            pass
        return ""

    # ──────────────────────────────────────────────────────────────
    # LOSPEC PALETTES
    # ──────────────────────────────────────────────────────────────

    def _search_lospec(self, name: str) -> str:
        """Suggest a Lospec palette based on the subject type."""
        name_lower = name.lower()

        palette_map = {
            "mage": "Endesga 32 — rich purples, magentas, deep blues for magic",
            "warrior": "Pear36 — warm metals, browns, burgundy for armor",
            "healer": "Arne 16 — soft whites, gold, gentle greens",
            "rogue": "Nyx8 — dark navy, charcoal, crimson highlights",
            "skeleton": "Rust Gold 8 — bone white, yellow-cream, dark shadow",
            "slime": "Slso8 — bright greens, cyan, translucent blues",
            "goblin": "Kirokaze Gameboy — dark green, olive, deep shadow",
            "dragon": "Aap-64 — deep obsidian, volcanic red, purple iridescence",
            "hero": "Pear36 — warm skin, bright primary colors, gold trim",
            "chest": "Slso8 — aged wood brown, gold clasp, shadow black",
            "castle": "Endesga 32 — stone grey, mossy green, torch orange",
            "forest": "Sweetie 16 — leaf greens, bark brown, sky blue",
            "dungeon": "Nyx8 — dark stone, torchlight amber, shadow black",
            "village": "Arne 16 — warm terracotta, timber brown, grass green",
        }

        for keyword, tip in palette_map.items():
            if keyword in name_lower:
                return tip

        return "Endesga 32 — versatile 32-color palette for game sprites"

    # ──────────────────────────────────────────────────────────────
    # STATIC TIPS (always included)
    # ──────────────────────────────────────────────────────────────

    def _static_tips(self, name: str, task: dict) -> str:
        w = task.get("width", 48)
        h = task.get("height", 48)
        name_lower = name.lower()

        tips = [f"Canvas {w}x{h}px"]

        if "character" in name_lower or any(k in name_lower for k in
                ["warrior", "mage", "rogue", "healer", "hero", "goblin", "skeleton"]):
            tips.append("Strong silhouette first — must be readable at 1x size")
            tips.append("Light source from top-left, shadow bottom-right")
            tips.append("3-4 colors per region max, 1px outline")

        elif "tile" in name_lower or "ground" in name_lower or "floor" in name_lower:
            tips.append("Seamless edges — top=bottom, left=right pixels must match")
            tips.append("Subtle noise texture, avoid large flat areas")

        elif "chest" in name_lower or "prop" in name_lower or "object" in name_lower:
            tips.append("Isometric-friendly shape, clear material read")
            tips.append("Rim light on top-left edge for 2.75D pop")

        elif "dragon" in name_lower or "boss" in name_lower:
            tips.append("Imposing silhouette — massive, fills canvas")
            tips.append("Scale texture with iridescence, glowing eyes as focal point")

        elif "slime" in name_lower:
            tips.append("Round jelly body, inner glow, highlight dot at top")

        return " | ".join(tips)
