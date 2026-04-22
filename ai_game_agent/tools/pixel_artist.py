"""
Pixel Art Engine — procedural pixel art generator.
Draws characters, tiles, objects, tilesets at top-game quality.
No external API — everything is generated locally with PIL + numpy.
"""
from __future__ import annotations
import math, random, json
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# ─── Master Palettes ──────────────────────────────────────────────────────────
# DB32 — industry standard 32-colour palette used in Stardew, Shovel Knight etc.
DB32 = [
    (0,0,0),(34,32,52),(69,40,60),(102,57,49),(143,86,59),(223,113,38),
    (217,160,102),(238,195,154),(251,242,54),(153,229,80),(106,190,48),
    (55,148,110),(75,105,47),(82,75,36),(50,60,57),(63,63,116),
    (48,96,130),(91,110,225),(99,155,255),(95,205,228),(203,219,252),
    (255,255,255),(155,173,183),(132,126,135),(105,106,106),(89,86,82),
    (118,66,138),(172,50,50),(217,87,99),(215,123,186),(143,151,74),
    (138,111,48),
]

# PICO-8 — 16 colours, retro-perfect
PICO8 = [
    (0,0,0),(29,43,83),(126,37,83),(0,135,81),(171,82,54),(95,87,79),
    (194,195,199),(255,241,232),(255,0,77),(255,163,0),(255,236,39),
    (0,228,54),(41,173,255),(131,118,156),(255,119,168),(255,204,170),
]

# Fantasy RPG warm palette (custom — village, isekai feel)
RPG_WARM = [
    (15,10,5),(40,25,15),(80,50,30),(120,75,45),(170,110,60),
    (210,160,90),(240,200,140),(255,235,180),(255,255,220),
    (30,60,20),(55,90,35),(85,130,50),(120,170,70),(165,210,100),
    (200,230,140),(40,80,120),(70,120,170),(110,165,210),(160,200,235),
    (200,225,250),(80,40,80),(130,60,120),(180,90,160),(220,140,200),
    (255,180,230),(180,50,40),(220,80,60),(255,120,90),(255,180,160),
    (200,80,30),(240,140,60),(255,200,100),
]

PALETTES = {"db32": DB32, "pico8": PICO8, "rpg": RPG_WARM}


def nearest_color(r: int, g: int, b: int, palette: list) -> tuple:
    best, dist = palette[0], float("inf")
    for c in palette:
        d = (r-c[0])**2 + (g-c[1])**2 + (b-c[2])**2
        if d < dist:
            dist, best = d, c
    return best


def quantize_image(img: Image.Image, palette_name: str = "db32") -> Image.Image:
    """Snap every pixel to the nearest palette colour."""
    pal = PALETTES.get(palette_name, DB32)
    arr = np.array(img.convert("RGBA"), dtype=np.int32)
    out = arr.copy()
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            if arr[y, x, 3] > 0:  # skip transparent
                r, g, b = int(arr[y, x, 0]), int(arr[y, x, 1]), int(arr[y, x, 2])
                c = nearest_color(r, g, b, pal)
                out[y, x, 0], out[y, x, 1], out[y, x, 2] = c
    return Image.fromarray(out.astype(np.uint8), "RGBA")


# ─── Character Generator ──────────────────────────────────────────────────────

class CharacterDrawer:
    """
    Draws a pixel art humanoid character at 16×16 or 32×32.
    Supports: warrior, mage, archer, villager, goblin, elf, skeleton.
    """

    ARCHETYPES = {
        "warrior":  {"hair": (80,50,30),  "skin": (215,160,100), "armor": (130,130,140), "eyes": (60,40,20),  "trim": (200,160,50)},
        "mage":     {"hair": (60,50,120), "skin": (220,180,140), "armor": (70,50,100),   "eyes": (120,180,255),"trim": (180,100,220)},
        "archer":   {"hair": (100,70,30), "skin": (200,150,90),  "armor": (60,90,50),    "eyes": (70,130,60), "trim": (150,100,40)},
        "villager": {"hair": (110,80,40), "skin": (230,185,140), "armor": (160,120,80),  "eyes": (80,60,30),  "trim": (190,150,100)},
        "goblin":   {"hair": (40,60,20),  "skin": (80,140,60),   "armor": (60,50,40),    "eyes": (255,200,0), "trim": (100,80,30)},
        "elf":      {"hair": (220,200,140),"skin": (200,230,180),"armor": (50,120,80),   "eyes": (100,200,150),"trim": (150,220,120)},
        "skeleton": {"hair": (200,190,160),"skin": (210,200,170),"armor": (180,170,140), "eyes": (255,80,40), "trim": (220,200,160)},
    }

    def __init__(self, archetype: str = "warrior", size: int = 32, palette: str = "db32", seed: int = 0):
        self.arch = self.ARCHETYPES.get(archetype, self.ARCHETYPES["warrior"])
        self.size = size
        self.palette_name = palette
        self.rng = random.Random(seed)

    def _scale(self, v: float) -> int:
        return max(1, round(v * self.size / 32))

    def draw(self, direction: str = "south") -> Image.Image:
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        s = self.size
        c = self.arch

        # Proportions
        head_h = s // 5
        head_w = s // 4
        head_x = s // 2 - head_w // 2
        head_y = self._scale(2)

        body_h = s // 3
        body_w = s // 3
        body_x = s // 2 - body_w // 2
        body_y = head_y + head_h

        leg_h = s // 4
        leg_w = self._scale(4)
        leg_y = body_y + body_h

        arm_h = body_h - self._scale(2)
        arm_w = self._scale(4)

        if direction in ("south", "north"):
            # ── Body ──
            d.rectangle([body_x, body_y, body_x+body_w, body_y+body_h], fill=c["armor"])
            # ── Legs ──
            d.rectangle([body_x+1, leg_y, body_x+leg_w, leg_y+leg_h], fill=c["armor"])
            d.rectangle([body_x+body_w-leg_w-1, leg_y, body_x+body_w-2, leg_y+leg_h], fill=c["armor"])
            # ── Arms ──
            d.rectangle([body_x-arm_w, body_y, body_x-1, body_y+arm_h], fill=c["armor"])
            d.rectangle([body_x+body_w+1, body_y, body_x+body_w+arm_w, body_y+arm_h], fill=c["armor"])
            # ── Hands ──
            hand_col = c["skin"]
            d.rectangle([body_x-arm_w, body_y+arm_h, body_x-1, body_y+arm_h+self._scale(3)], fill=hand_col)
            d.rectangle([body_x+body_w+1, body_y+arm_h, body_x+body_w+arm_w, body_y+arm_h+self._scale(3)], fill=hand_col)
            # ── Trim ──
            d.rectangle([body_x, body_y, body_x+body_w, body_y+self._scale(2)], fill=c["trim"])
            # ── Head ──
            d.rectangle([head_x, head_y, head_x+head_w, head_y+head_h], fill=c["skin"])
            # ── Hair ──
            hair_h = self._scale(4) if direction == "south" else self._scale(3)
            d.rectangle([head_x, head_y, head_x+head_w, head_y+hair_h], fill=c["hair"])
            # ── Eyes (south only) ──
            if direction == "south":
                ey = head_y + head_h - self._scale(4)
                ex1 = head_x + self._scale(3)
                ex2 = head_x + head_w - self._scale(4)
                d.point([(ex1, ey)], fill=c["eyes"])
                d.point([(ex2, ey)], fill=c["eyes"])
            # ── Feet ──
            d.rectangle([body_x+1, leg_y+leg_h, body_x+leg_w, leg_y+leg_h+self._scale(2)], fill=(40,30,20))
            d.rectangle([body_x+body_w-leg_w-1, leg_y+leg_h, body_x+body_w-2, leg_y+leg_h+self._scale(2)], fill=(40,30,20))

        elif direction in ("east", "west"):
            flip = direction == "west"
            # Body side view
            d.rectangle([body_x, body_y, body_x+body_w-self._scale(4), body_y+body_h], fill=c["armor"])
            d.rectangle([body_x, leg_y, body_x+leg_w, leg_y+leg_h], fill=c["armor"])
            d.rectangle([body_x+self._scale(6), leg_y+self._scale(4), body_x+leg_w+self._scale(6), leg_y+leg_h], fill=c["armor"])
            # Arm
            arm_x = body_x + body_w - self._scale(4) if not flip else body_x - arm_w
            d.rectangle([arm_x, body_y, arm_x+arm_w, body_y+arm_h], fill=c["armor"])
            d.rectangle([arm_x, body_y+arm_h, arm_x+arm_w, body_y+arm_h+self._scale(3)], fill=c["skin"])
            # Head
            d.rectangle([head_x, head_y, head_x+head_w, head_y+head_h], fill=c["skin"])
            d.rectangle([head_x, head_y, head_x+head_w, head_y+self._scale(4)], fill=c["hair"])
            # Nose
            nose_x = head_x + head_w if not flip else head_x - 1
            d.point([(nose_x, head_y + head_h - self._scale(4))], fill=c["skin"])
            # Eye
            ey = head_y + head_h - self._scale(5)
            ex = head_x + head_w - self._scale(3) if not flip else head_x + self._scale(2)
            d.point([(ex, ey)], fill=c["eyes"])
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return quantize_image(img, self.palette_name)


# ─── Tile Generator ───────────────────────────────────────────────────────────

class TileDrawer:
    """
    Draws individual 16×16 or 32×32 pixel art tiles.
    Types: grass, dirt, stone, water, sand, forest_floor, lava, snow, swamp.
    """

    TILE_PALETTES = {
        "grass":        [(55,148,110),(106,190,48),(82,124,36),(55,90,25),(200,230,140)],
        "dirt":         [(143,86,59),(102,57,49),(80,50,35),(160,110,70),(180,140,90)],
        "stone":        [(95,87,79),(132,126,135),(155,173,183),(70,70,70),(200,200,200)],
        "water":        [(48,96,130),(63,63,116),(91,110,225),(160,200,240),(220,240,255)],
        "sand":         [(217,160,102),(238,195,154),(200,170,90),(180,150,70),(240,215,160)],
        "forest_floor": [(40,60,20),(55,80,30),(70,100,40),(35,50,15),(90,120,50)],
        "lava":         [(180,50,10),(220,80,20),(255,120,40),(255,180,60),(240,100,30)],
        "snow":         [(200,220,240),(220,235,250),(240,245,255),(180,200,220),(255,255,255)],
        "swamp":        [(40,60,30),(60,80,40),(80,100,50),(30,50,25),(100,120,60)],
        "wood":         [(138,111,48),(120,90,40),(100,70,30),(160,130,70),(180,150,90)],
        "brick":        [(143,86,59),(120,70,45),(160,100,65),(100,55,35),(180,120,80)],
    }

    def __init__(self, tile_type: str = "grass", size: int = 16, seed: int = 0):
        self.tile_type = tile_type
        self.size = size
        self.rng = random.Random(seed)
        self.colours = self.TILE_PALETTES.get(tile_type, self.TILE_PALETTES["grass"])

    def draw(self) -> Image.Image:
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 255))
        arr = np.array(img)
        s = self.size
        base = self.colours[0]
        arr[:, :, 0] = base[0]
        arr[:, :, 1] = base[1]
        arr[:, :, 2] = base[2]

        if self.tile_type in ("grass", "forest_floor", "swamp"):
            self._draw_organic(arr)
        elif self.tile_type == "water":
            self._draw_water(arr)
        elif self.tile_type in ("stone", "brick"):
            self._draw_stone(arr)
        elif self.tile_type in ("dirt", "sand"):
            self._draw_dirt(arr)
        elif self.tile_type == "lava":
            self._draw_lava(arr)
        elif self.tile_type == "snow":
            self._draw_snow(arr)
        elif self.tile_type == "wood":
            self._draw_wood(arr)
        else:
            self._draw_organic(arr)

        img = Image.fromarray(arr.astype(np.uint8), "RGBA")
        return quantize_image(img, "db32")

    def _draw_organic(self, arr):
        s = self.size
        c = self.colours
        for y in range(s):
            for x in range(s):
                n = self.rng.random()
                if n < 0.55:
                    col = c[0]
                elif n < 0.75:
                    col = c[1]
                elif n < 0.88:
                    col = c[2]
                elif n < 0.95:
                    col = c[3]
                else:
                    col = c[4]
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = col

    def _draw_water(self, arr):
        s = self.size
        c = self.colours
        for y in range(s):
            for x in range(s):
                wave = math.sin((x + y) * 0.8 + self.rng.uniform(0, 0.3)) * 0.5 + 0.5
                idx = min(int(wave * len(c)), len(c)-1)
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = c[idx]

    def _draw_stone(self, arr):
        s = self.size
        c = self.colours
        brick_h = max(4, s // 4)
        brick_w = max(6, s // 2)
        for y in range(s):
            row = y // brick_h
            offset = (brick_w // 2) if (row % 2 == 1) else 0
            for x in range(s):
                bx = (x + offset) % brick_w
                by = y % brick_h
                is_edge = (bx == 0 or by == 0)
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = c[0] if is_edge else c[1+self.rng.randint(0,2)]

    def _draw_dirt(self, arr):
        s = self.size
        c = self.colours
        for y in range(s):
            for x in range(s):
                n = self.rng.random()
                col = c[0] if n < 0.5 else (c[1] if n < 0.8 else c[2])
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = col
                if self.rng.random() < 0.06:
                    arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = c[3]

    def _draw_lava(self, arr):
        s = self.size
        c = self.colours
        for y in range(s):
            for x in range(s):
                glow = math.sin(x * 0.7) * math.cos(y * 0.7) * 0.5 + 0.5
                idx = min(int(glow * len(c)), len(c)-1)
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = c[idx]

    def _draw_snow(self, arr):
        s = self.size
        c = self.colours
        for y in range(s):
            for x in range(s):
                n = self.rng.random()
                col = c[4] if n > 0.85 else (c[0] if n > 0.4 else c[1])
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = col

    def _draw_wood(self, arr):
        s = self.size
        c = self.colours
        for y in range(s):
            grain = math.sin(y * 1.5 + self.rng.uniform(-0.5,0.5)) * 0.5 + 0.5
            for x in range(s):
                idx = min(int(grain * len(c)), len(c)-1)
                arr[y, x, 0], arr[y, x, 1], arr[y, x, 2] = c[idx]


# ─── Object / Prop Generator ──────────────────────────────────────────────────

class PropDrawer:
    """Draws game props: tree, chest, door, sign, barrel, torch, flower."""

    def draw(self, prop_type: str = "tree", size: int = 32, seed: int = 0) -> Image.Image:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        fn = getattr(self, f"_draw_{prop_type}", self._draw_tree)
        fn(d, size, random.Random(seed))
        return quantize_image(img, "db32")

    def _draw_tree(self, d, s, rng):
        # Trunk
        tw = max(4, s // 8)
        tx = s // 2 - tw // 2
        ty = s * 3 // 5
        d.rectangle([tx, ty, tx+tw, s-2], fill=(100,70,30))
        # Canopy — 3 overlapping circles
        gr = (55, 148, 110)
        gd = (40, 100, 60)
        gl = (106, 190, 48)
        r = s // 3
        cx, cy = s//2, s//3
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=gr)
        d.ellipse([cx-r+2, cy-r-4, cx+r-2, cy+r-4], fill=gl)
        d.ellipse([cx-r//2, cy-r//2+4, cx+r//2, cy+r//2+4], fill=gd)

    def _draw_chest(self, d, s, rng):
        cx, cy = 2, s//2
        cw, ch = s-4, s//3
        d.rectangle([cx, cy, cx+cw, cy+ch], fill=(138,111,48))
        d.rectangle([cx, cy, cx+cw, cy+2], fill=(100,80,30))
        d.rectangle([cx, cy+ch//2-1, cx+cw, cy+ch//2+1], fill=(200,160,50))
        d.rectangle([cx+cw//2-2, cy+ch//2-3, cx+cw//2+2, cy+ch//2+3], fill=(220,180,60))

    def _draw_torch(self, d, s, rng):
        tw = max(3, s//10)
        tx = s//2 - tw//2
        d.rectangle([tx, s//2, tx+tw, s-4], fill=(100,70,30))
        d.ellipse([tx-2, s//4, tx+tw+2, s//2+4], fill=(220,80,20))
        d.ellipse([tx, s//4+2, tx+tw, s//2+2], fill=(255,180,40))
        d.ellipse([tx+1, s//4+4, tx+tw-1, s//2], fill=(255,240,100))

    def _draw_barrel(self, d, s, rng):
        bx, bw, bh = s//4, s//2, s*2//3
        by = s - bh - 2
        d.ellipse([bx, by, bx+bw, by+bh], fill=(138,111,48))
        for band_y in [by+bh//4, by+bh//2, by+3*bh//4]:
            d.rectangle([bx, band_y, bx+bw, band_y+1], fill=(80,60,30))

    def _draw_sign(self, d, s, rng):
        d.rectangle([s//2-1, s//2, s//2+1, s-4], fill=(120,90,40))
        d.rectangle([4, s//4, s-4, s//2+2], fill=(160,120,60))
        d.rectangle([5, s//4+1, s-5, s//4+2], fill=(200,160,90))

    def _draw_flower(self, d, s, rng):
        cx, cy = s//2, s//2
        colours = [(255,80,120),(255,200,40),(200,80,255),(80,180,255)]
        col = rng.choice(colours)
        r = max(3, s//6)
        for angle in range(0, 360, 60):
            px = cx + int(math.cos(math.radians(angle)) * r * 1.5)
            py = cy + int(math.sin(math.radians(angle)) * r * 1.5)
            d.ellipse([px-r//2, py-r//2, px+r//2, py+r//2], fill=col)
        d.ellipse([cx-r//2, cy-r//2, cx+r//2, cy+r//2], fill=(255,240,80))
        stem_y = cy + r
        d.rectangle([cx-1, stem_y, cx+1, s-2], fill=(55,148,110))


# ─── Public API ───────────────────────────────────────────────────────────────

def draw_character(
    archetype: str = "warrior",
    size: int = 32,
    direction: str = "south",
    palette: str = "db32",
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Draw a pixel art character and optionally save it."""
    img = CharacterDrawer(archetype, size, palette, seed).draw(direction)
    if output_path:
        img.save(output_path)
    return img


def draw_tile(
    tile_type: str = "grass",
    size: int = 16,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Image.Image:
    img = TileDrawer(tile_type, size, seed).draw()
    if output_path:
        img.save(output_path)
    return img


def draw_prop(
    prop_type: str = "tree",
    size: int = 32,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> Image.Image:
    img = PropDrawer().draw(prop_type, size, seed)
    if output_path:
        img.save(output_path)
    return img


def draw_to_base64(img: Image.Image) -> str:
    import io, base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def draw_character_all_directions(
    archetype: str = "warrior",
    size: int = 32,
    palette: str = "db32",
    seed: int = 0,
    output_dir: Optional[str] = None,
) -> dict[str, Image.Image]:
    """Draw all 4 directions and return dict."""
    directions = ["south", "north", "east", "west"]
    drawer = CharacterDrawer(archetype, size, palette, seed)
    result = {}
    for direction in directions:
        img = drawer.draw(direction)
        result[direction] = img
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            img.save(f"{output_dir}/{archetype}_{direction}.png")
    return result
