"""
Reference Library — AI Dev's art school.

Downloads and indexes CC0 pixel art + anime-style reference images.
Uses Gemini Vision to analyze each image and extract technique lessons
that AI Dev injects into every drawing session.
"""
from __future__ import annotations
import base64
import json
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# macOS SSL fix
_ssl_ctx = ssl.create_default_context()
try:
    import certifi
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

# ── CC0 Pixel Art reference images (OpenGameArt / Kenney / public domain) ──
PIXEL_ART_REFS = [
    # Kenney.nl — reliable CDN URLs (CC0)
    ("kenney_tiles_1",    "https://kenney.nl/content/3-assets/49-tiny-town/preview.png"),
    ("kenney_rpg",        "https://kenney.nl/content/3-assets/56-rpg-pack/preview.png"),
    # OpenGameArt CC0 previews
    ("lpc_terrain",       "https://opengameart.org/sites/default/files/styles/medium/public/terrain_tileset.png"),
    ("cave_tiles",        "https://opengameart.org/sites/default/files/styles/medium/public/dungeonTilesetII_v300.png"),
    # Lospec public pixel art palettes (always online)
    ("pixel_landscape",   "https://lospec.com/palette-list/vinik24.png"),
    ("pixel_forest",      "https://lospec.com/palette-list/endesga-32.png"),
]

# ── Anime-style art technique guides (public domain / art education) ──
ANIME_REFS = [
    # Wikimedia Commons — stable public domain art references
    ("anime_palette_study", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/200px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"),
    ("anime_bg_depth",      "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/200px-Camponotus_flavomarginatus_ant.jpg"),
]

# ── Embedded technique lessons (AI Dev learns these without needing images) ──
BUILT_IN_LESSONS = {
    "2.75D_depth": {
        "technique": "2.75D Depth Illusion",
        "rules": [
            "Background layer: desaturated, slightly blue-shifted, minimal detail",
            "Midground layer: normal saturation, medium detail, slightly smaller scale",
            "Foreground layer: high saturation, maximum detail, largest scale",
            "Use atmospheric perspective: far objects are lighter and cooler in hue",
            "Parallax hints: overlap elements so near occludes far",
            "Cast shadows fall toward camera (downward in top-3/4 view)",
            "Light source consistent across ALL layers (top-left is classic)",
        ],
        "palette_rules": [
            "Limit to 16-32 colors per scene",
            "Each layer has 4-8 colors max",
            "Dark value for shadows: base color × 0.55",
            "Light value for highlights: base color + (white × 0.35)",
            "Background is 40% desaturated version of midground palette",
        ],
        "examples": "Octopath Traveler, Dead Cells, Sea of Stars, CrossCode"
    },
    "anime_pixel_style": {
        "technique": "Anime-influenced Pixel Art",
        "rules": [
            "Strong, clean 1px outlines — pure black (#000000) or very dark shade of fill color",
            "Cel-shaded: flat base + 1 shadow + 1 highlight, no gradients on characters",
            "Hair: distinct highlight blob, dark shadow underneath",
            "Eyes: large iris, white specular dot (1px), dark pupil",
            "Skin: warm base (#f5c5a3), shadow (#d4956f), highlight (#fde8d0)",
            "Clothes: saturated colors, sharp shadow lines following form",
            "Anime-style proportions: larger eyes (3-4px tall on 32px sprite), smaller mouth (1-2px)",
            "Use color ramps not dithering for character shading",
        ],
        "palette_rules": [
            "Skin ramp: #fde8d0 → #f5c5a3 → #d4956f → #a0634a",
            "Hair often uses 3 tones: highlight, base, shadow",
            "Saturated outfits, complementary palette pairs",
            "Background significantly less saturated than character",
        ],
        "examples": "Owlboy, Cave Story, Undertale, Iconoclasts"
    },
    "goty_polish": {
        "technique": "Game of the Year Polish Standards",
        "rules": [
            "Every sprite has a clear silhouette readable at thumbnail size",
            "Color count discipline: never exceed 16 colors per sprite",
            "Sub-pixel animation: even static sprites have implied motion in their pose",
            "Consistent pixel size: NEVER mix 1px and 2px detail on same element",
            "Anti-aliased edges in pixel art are WRONG — use clean integer coordinates only",
            "Each object casts a visible shadow, shadows match light source angle",
            "Background tiles must be tileable (edge matches edge perfectly)",
            "Foreground objects always have stronger value contrast than background",
            "Use rim lighting for depth: thin highlight on dark-side edge",
            "Dithering for gradients in backgrounds only, not on characters",
        ],
        "references": "Celeste, Shovel Knight, Hollow Knight, Stardew Valley, Terraria"
    },
    "background_layers": {
        "technique": "Professional Background Construction",
        "rules": [
            "Sky/void layer: simple gradient, 2-3 colors, fullwidth",
            "Far background (30% screen): silhouettes only, 2-3 dark values",
            "Mid background (50% screen): recognizable shapes, 4-6 colors",
            "Near background (20% screen): full detail, darkest values at base",
            "Ground plane: must read as horizontal surface (horizontal lines/texture)",
            "Atmospheric haze: lighter, bluer rows near horizon",
            "Add parallax markers: distinct shapes at each depth that wont overlap",
        ]
    }
}


class ReferenceLibrary:
    """AI Dev's art school — manages references and lessons."""

    # Search queries AI Dev uses to find pixel art references
    SEARCH_QUERIES = [
        "pixel art game tiles sprite sheet RPG",
        "pixel art character sprite animation sheet",
        "pixel art background scene environment 2D game",
        "pixel art dungeon tileset indie game",
        "anime pixel art scene landscape",
        "pixel art prop item game asset",
        "2D platformer pixel art environment",
        "pixel art UI HUD game interface",
    ]

    def __init__(
        self,
        ref_dir: str = "training_data/references",
        lesson_file: str = "training_data/references/lessons.json",
    ):
        self.ref_dir = Path(ref_dir)
        self.pixel_dir = self.ref_dir / "pixel_art"
        self.anime_dir = self.ref_dir / "anime"
        self.search_dir = self.ref_dir / "searched"
        self.lesson_file = Path(lesson_file)
        self.pixel_dir.mkdir(parents=True, exist_ok=True)
        self.anime_dir.mkdir(parents=True, exist_ok=True)
        self.search_dir.mkdir(parents=True, exist_ok=True)
        self._lessons: Dict[str, dict] = dict(BUILT_IN_LESSONS)
        self._load_saved_lessons()

    # ── Image search (DuckDuckGo — free, no API key) ─────────────────────

    def search_pixel_art_images(
        self,
        queries: Optional[List[str]] = None,
        images_per_query: int = 3,
        verbose: bool = True,
    ) -> int:
        """Search for pixel art images via DuckDuckGo and download them."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            if verbose:
                print("  ⚠️  duckduckgo-search not installed. Run: pip3 install duckduckgo-search")
            return 0

        queries = queries or self.SEARCH_QUERIES
        downloaded = 0

        if verbose:
            print(f"\n🔍 Searching for pixel art references ({len(queries)} queries)...")

        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = list(ddgs.images(
                        keywords=query,
                        max_results=images_per_query,
                        size="Small",       # pixel art is usually small
                        type_image="photo",
                    ))
                    for i, r in enumerate(results):
                        url = r.get("image", "")
                        if not url:
                            continue
                        # Safe filename from query
                        safe = re.sub(r"[^a-z0-9]+", "_", query.lower())[:40]
                        fname = f"{safe}_{i}.png"
                        path = self.search_dir / fname
                        if path.exists():
                            continue
                        try:
                            req = urllib.request.Request(
                                url, headers={"User-Agent": "Mozilla/5.0"}
                            )
                            with urllib.request.urlopen(req, timeout=10, context=_ssl_ctx) as r:
                                data = r.read()
                            # Only save actual image data (skip tiny icons < 1KB)
                            if len(data) > 1000:
                                path.write_bytes(data)
                                downloaded += 1
                                if verbose:
                                    print(f"  ✅ {fname} ({len(data)//1024}KB)")
                        except Exception:
                            pass  # silently skip broken image URLs
                    time.sleep(1.0)  # be polite between queries
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Search failed for '{query[:40]}': {e}")
                    continue

        if verbose:
            print(f"  📚 Downloaded {downloaded} reference images → {self.search_dir}")
        return downloaded

    # ── Downloading references ────────────────────────────────────────────

    def download_references(self, verbose: bool = True) -> int:
        """Download CC0 reference images. Returns count downloaded."""
        downloaded = 0
        for name, url in PIXEL_ART_REFS:
            path = self.pixel_dir / f"{name}.png"
            if not path.exists():
                try:
                    if verbose:
                        print(f"  📥 Downloading pixel art ref: {name}")
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as r:
                        path.write_bytes(r.read())
                    downloaded += 1
                    time.sleep(0.5)
                except Exception as e:
                    if verbose:
                        print(f"     ⚠️ Failed {name}: {e}")

        for name, url in ANIME_REFS:
            path = self.anime_dir / f"{name}.png"
            if not path.exists():
                try:
                    if verbose:
                        print(f"  📥 Downloading anime ref: {name}")
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=15, context=_ssl_ctx) as r:
                        data = r.read()
                    path.write_bytes(data)
                    downloaded += 1
                    time.sleep(0.5)
                except Exception as e:
                    if verbose:
                        print(f"     ⚠️ Failed {name}: {e}")

        return downloaded

    def index_user_references(self) -> List[Path]:
        """Find all reference images (user-placed + downloaded + searched)."""
        images = []
        for folder in [self.pixel_dir, self.anime_dir, self.search_dir]:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                images.extend(folder.glob(ext))
        return sorted(images)

    # ── Gemini Vision analysis ────────────────────────────────────────────

    def analyze_reference(self, image_path: Path, api_key: str) -> Optional[dict]:
        """Use Gemini Vision to analyze a reference image and extract pixel art lessons."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            img_bytes = image_path.read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode()

            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            prompt = """Analyze this pixel art image as a master pixel art teacher.
Extract SPECIFIC, ACTIONABLE technique lessons an AI student can immediately apply.

Return JSON with this exact structure:
{
  "technique": "brief name",
  "style_category": "pixel_art | anime_inspired | background | character | tileset | prop",
  "color_palette": ["#hex1", "#hex2", ...],  // up to 12 dominant colors
  "palette_strategy": "describe how colors are organized (ramps, complementary, etc.)",
  "shading_method": "flat | 2-tone | 3-tone | dithered | gradient",
  "outline_style": "none | 1px-black | 1px-dark | selective",
  "depth_layers": true/false,  // does it use visible depth planes?
  "detail_density": "sparse | medium | dense",
  "light_source": "top-left | top | top-right | none",
  "pixel_size": "1px | 2px | mixed",
  "key_techniques": ["specific technique 1", "technique 2", ...],  // up to 6
  "what_makes_it_great": "specific quality observations",
  "drawing_instructions": ["Step 1: ...", "Step 2: ...", ...]  // 4-6 steps to recreate this style
}
Return ONLY valid JSON, no markdown."""

            import PIL.Image as PILImage
            import io
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            # Resize large refs to save tokens
            if pil_img.width > 512 or pil_img.height > 512:
                pil_img.thumbnail((512, 512), PILImage.LANCZOS)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_b64}
            ])
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            lesson = json.loads(raw)
            lesson["source_image"] = str(image_path.name)
            return lesson

        except Exception as e:
            print(f"  ⚠️ Analysis failed for {image_path.name}: {e}")
            return None

    def analyze_all_references(self, api_key: str, force: bool = False) -> int:
        """Analyze all reference images not yet analyzed. Returns count added."""
        images = self.index_user_references()
        added = 0
        for img_path in images:
            key = img_path.stem
            if key in self._lessons and not force:
                continue
            print(f"  🔍 Studying: {img_path.name}")
            lesson = self.analyze_reference(img_path, api_key)
            if lesson:
                self._lessons[key] = lesson
                added += 1
                print(f"     ✅ Learned: {lesson.get('technique', key)}")
            time.sleep(1)  # Rate limit
        self._save_lessons()
        return added

    # ── Lesson retrieval ──────────────────────────────────────────────────

    def get_lessons_for_task(self, task_type: str, n: int = 4) -> List[dict]:
        """
        Return the most relevant lessons for a given drawing task.
        task_type: 'background', 'character', 'tileset', 'prop', 'full_scene'
        """
        # Always include core quality lessons
        core = ["2.75D_depth", "goty_polish", "anime_pixel_style", "background_layers"]
        selected = [self._lessons[k] for k in core if k in self._lessons]

        # Add image-derived lessons matching task type
        for key, lesson in self._lessons.items():
            if key in core:
                continue
            cat = lesson.get("style_category", "")
            if task_type in cat or cat in task_type:
                selected.append(lesson)
            if len(selected) >= n + len(core):
                break

        return selected[:n + len(core)]

    def lesson_summary(self) -> str:
        """Compact summary of all lessons suitable for injection into a prompt."""
        lines = ["=== AI DEV LEARNED TECHNIQUES ===\n"]
        for key, lesson in self._lessons.items():
            name = lesson.get("technique", key)
            lines.append(f"### {name}")
            if "rules" in lesson:
                for r in lesson["rules"][:6]:
                    lines.append(f"  • {r}")
            if "key_techniques" in lesson:
                for t in lesson["key_techniques"][:4]:
                    lines.append(f"  • {t}")
            if "palette_rules" in lesson:
                for p in lesson["palette_rules"][:4]:
                    lines.append(f"  • {p}")
            if "drawing_instructions" in lesson:
                for i, step in enumerate(lesson["drawing_instructions"][:5], 1):
                    lines.append(f"  Step {i}: {step}")
            if "color_palette" in lesson:
                lines.append(f"  Palette: {' '.join(lesson['color_palette'][:8])}")
            lines.append("")
        return "\n".join(lines)

    def lesson_count(self) -> int:
        return len(self._lessons)

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_saved_lessons(self) -> None:
        if self.lesson_file.exists():
            try:
                saved = json.loads(self.lesson_file.read_text())
                self._lessons.update(saved)
            except Exception:
                pass

    def _save_lessons(self) -> None:
        self.lesson_file.parent.mkdir(parents=True, exist_ok=True)
        # Don't save built-in lessons (they're always in code)
        to_save = {k: v for k, v in self._lessons.items() if k not in BUILT_IN_LESSONS}
        self.lesson_file.write_text(json.dumps(to_save, indent=2))
