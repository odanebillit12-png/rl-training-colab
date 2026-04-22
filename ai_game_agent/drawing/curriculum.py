"""
Drawing Curriculum — AI Dev's game dev art school progression.

6 levels from absolute beginner to GOTY master.
Every task produces a real game asset: tiles, characters, UI, scenes.
AI Dev must hit the min_score to advance to the next level.
"""
from typing import List, Dict, Any

DRAWING_CURRICULUM: List[Dict[str, Any]] = [
    {
        "level": 1,
        "name": "Game Tiles Basics",
        "description": "Draw the fundamental tileset pieces every 2D game needs. 6-color palette max. Clean edges.",
        "min_score": 0,
        "graduate_at": 45,
        "canvas_size": (32, 32),
        "max_commands": 150,
        "tasks": [
            {
                "task_name": "grass_tile",
                "description": "a seamless grass ground tile for a 2D RPG — bright green base, 2 shadow shades, tiny blade details at top edge",
                "style": "game tileset tile, top-down RPG",
                "width": 32, "height": 32,
                "max_commands": 100,
            },
            {
                "task_name": "stone_floor_tile",
                "description": "a seamless stone floor tile — grey stone slabs, mortar lines, 3 shades of grey, subtle wear marks",
                "style": "game tileset tile, dungeon floor",
                "width": 32, "height": 32,
                "max_commands": 120,
            },
            {
                "task_name": "water_tile",
                "description": "a seamless animated-style water tile — 3 shades of blue, simple ripple lines suggesting movement",
                "style": "game tileset tile, water surface",
                "width": 32, "height": 32,
                "max_commands": 120,
            },
            {
                "task_name": "dirt_path_tile",
                "description": "a seamless dirt path tile for an RPG — warm brown base, pebble details, lighter center worn path",
                "style": "game tileset tile, outdoor path",
                "width": 32, "height": 32,
                "max_commands": 100,
            },
        ],
    },
    {
        "level": 2,
        "name": "Game Props & Objects",
        "description": "Draw interactive game objects with 3-tone shading: base, shadow, highlight. Clean black outlines.",
        "min_score": 45,
        "graduate_at": 58,
        "canvas_size": (48, 48),
        "max_commands": 250,
        "tasks": [
            {
                "task_name": "treasure_chest",
                "description": "a pixel art treasure chest game prop — wood planks, metal clasp and hinges, top-left light source, 3-tone wood shading, clean 1px black outline",
                "style": "pixel art RPG game prop, cel-shaded",
                "width": 48, "height": 48,
                "max_commands": 200,
            },
            {
                "task_name": "health_potion",
                "description": "a red health potion bottle game item — glass bottle shape, red liquid, cork stopper, glint highlight on glass, drop shadow",
                "style": "pixel art RPG item icon",
                "width": 48, "height": 48,
                "max_commands": 180,
            },
            {
                "task_name": "wall_torch",
                "description": "a wall-mounted torch game prop — iron bracket, wooden handle, warm orange/yellow flame, orange glow on wall behind, 1px outline",
                "style": "pixel art dungeon prop with light emission",
                "width": 48, "height": 48,
                "max_commands": 200,
            },
            {
                "task_name": "wooden_barrel",
                "description": "a wooden barrel prop — curved wood planks with vertical grain lines, metal hoops, top-left light, dark shadow underneath",
                "style": "pixel art game prop, 3-tone shading",
                "width": 48, "height": 48,
                "max_commands": 200,
            },
        ],
    },
    {
        "level": 3,
        "name": "Game Characters",
        "description": "Draw playable characters and enemies. Clean silhouettes, anime-pixel cel shading, readable at small size.",
        "min_score": 58,
        "graduate_at": 68,
        "canvas_size": (48, 64),
        "max_commands": 350,
        "tasks": [
            {
                "task_name": "hero_warrior",
                "description": "a pixel art RPG hero warrior sprite facing south — sword at side, shield, fantasy armor, strong readable silhouette, anime-pixel cel shading, 3-tone shading on every surface",
                "style": "anime-influenced pixel art player character, top-down 3/4 view",
                "width": 48, "height": 64,
                "max_commands": 350,
            },
            {
                "task_name": "mage_character",
                "description": "a pixel art mage character — blue/purple robes, pointed hat with star, glowing magic staff, flowing cloak, anime style, cel shaded",
                "style": "anime pixel art mage character sprite",
                "width": 48, "height": 64,
                "max_commands": 350,
            },
            {
                "task_name": "slime_enemy",
                "description": "a pixel art slime enemy sprite — round gelatinous body, cute anime-style eyes, shiny reflective highlight on top, dark shadow at base, 2 outline colors",
                "style": "anime pixel art enemy, cute-creepy style",
                "width": 48, "height": 48,
                "max_commands": 250,
            },
            {
                "task_name": "skeleton_enemy",
                "description": "a pixel art skeleton enemy — bone armor, glowing eye sockets, cracked bones, dark fantasy style, strong contrast, readable silhouette",
                "style": "pixel art dark fantasy enemy sprite",
                "width": 48, "height": 64,
                "max_commands": 350,
            },
        ],
    },
    {
        "level": 4,
        "name": "Game UI & HUD",
        "description": "Draw game interface elements: health bars, inventory slots, buttons, dialogue boxes.",
        "min_score": 68,
        "graduate_at": 78,
        "canvas_size": (96, 64),
        "max_commands": 500,
        "tasks": [
            {
                "task_name": "hud_health_bar",
                "description": (
                    "a pixel art RPG HUD panel: "
                    "red heart health bar (10 hearts, some full some empty), "
                    "blue mana bar, "
                    "gold coin counter with coin icon, "
                    "dark semi-transparent panel background, "
                    "fantasy styled pixel font numbers, "
                    "polished UI like Celeste or Hollow Knight"
                ),
                "style": "pixel art game HUD, fantasy RPG style",
                "width": 96, "height": 32,
                "max_commands": 400,
            },
            {
                "task_name": "inventory_grid",
                "description": (
                    "a pixel art inventory grid UI — "
                    "4x3 grid of item slots, dark stone-textured background, "
                    "golden border on selected slot, "
                    "3 slots showing items (sword, potion, key), "
                    "others empty with subtle inset shadow"
                ),
                "style": "pixel art RPG inventory UI",
                "width": 96, "height": 64,
                "max_commands": 450,
            },
            {
                "task_name": "dialogue_box",
                "description": (
                    "a pixel art RPG dialogue box at bottom of screen — "
                    "dark parchment/stone texture background, "
                    "decorative corner ornaments, "
                    "2-line text area with cursor blink indicator, "
                    "speaker portrait frame on left side, "
                    "polished fantasy RPG style"
                ),
                "style": "pixel art RPG dialogue UI element",
                "width": 96, "height": 32,
                "max_commands": 350,
            },
            {
                "task_name": "skill_buttons",
                "description": (
                    "4 pixel art ability/skill buttons in a row — "
                    "fire spell (red), ice spell (blue), lightning (yellow), heal (green), "
                    "each with icon, hotkey number, cooldown ring, "
                    "fantasy game action bar style"
                ),
                "style": "pixel art game ability bar UI",
                "width": 96, "height": 32,
                "max_commands": 400,
            },
        ],
    },
    {
        "level": 5,
        "name": "Game Scenes & Levels",
        "description": "Draw complete game level backgrounds with 2.75D depth. Ready to use in Godot as a scene backdrop.",
        "min_score": 78,
        "graduate_at": 88,
        "canvas_size": (128, 96),
        "max_commands": 700,
        "tasks": [
            {
                "task_name": "rpg_overworld",
                "description": (
                    "a 2.75D pixel art RPG overworld scene: "
                    "background=soft sky with distant mountains (desaturated blue-purple), "
                    "midground=green rolling hills, a winding dirt road, distant village rooftops, "
                    "foreground=detailed grass, flowers, stone well, fence posts, "
                    "warm daylight from top-left, anime-influenced palette, "
                    "polished to Octopath Traveler / Sea of Stars standard"
                ),
                "style": "2.75D layered pixel art RPG overworld, anime-inspired",
                "width": 128, "height": 96,
                "max_commands": 700,
            },
            {
                "task_name": "dungeon_level",
                "description": (
                    "a 2.75D pixel art dungeon level scene: "
                    "background=deep dark void with faint torch glow on distant walls, "
                    "midground=stone brick walls, iron door, hanging chains, "
                    "foreground=cracked stone floor, puddles of water reflecting torchlight, "
                    "Dead Cells / Hollow Knight quality darkness and detail"
                ),
                "style": "2.75D layered pixel art dungeon, dark fantasy",
                "width": 128, "height": 96,
                "max_commands": 700,
            },
            {
                "task_name": "village_marketplace",
                "description": (
                    "a 2.75D pixel art isekai village marketplace: "
                    "background=clear blue sky with clouds, distant castle tower, "
                    "midground=market stalls with colorful awnings, NPC silhouettes, cobblestone street, "
                    "foreground=barrel, crates, flowers in pots, signpost, "
                    "warm noon sunlight, anime-influenced fantasy palette"
                ),
                "style": "2.75D isekai RPG village scene, anime pixel art",
                "width": 128, "height": 96,
                "max_commands": 700,
            },
        ],
    },
    {
        "level": 6,
        "name": "GOTY Game Master",
        "description": "Create complete game-ready art at award-winning quality. Every asset must score 88+.",
        "min_score": 88,
        "graduate_at": 95,
        "canvas_size": (192, 128),
        "max_commands": 1000,
        "tasks": [
            {
                "task_name": "title_screen",
                "description": (
                    "a GOTY-quality pixel art game title screen: "
                    "epic isekai RPG world — hero silhouette on cliff edge against massive setting sun, "
                    "floating islands in the distance, ancient dragon skeleton arch overhead, "
                    "5 depth planes of atmospheric perspective, "
                    "anime-quality color grading, every pixel intentional, "
                    "better than Celeste, Hollow Knight, and Dead Cells combined"
                ),
                "style": "GOTY pixel art title screen, cinematic game art",
                "width": 192, "height": 128,
                "max_commands": 1000,
            },
            {
                "task_name": "boss_arena",
                "description": (
                    "a GOTY-quality pixel art boss battle arena: "
                    "ancient ruins under a stormy sky, cracked stone floor with glowing ritual circle, "
                    "massive stone pillars with carved runes, "
                    "dramatic lighting from the ritual circle below and lightning above, "
                    "particle/magic effects suggested through dithering, "
                    "Dead Cells / Elden Ring quality atmosphere in pixel art"
                ),
                "style": "GOTY pixel art boss arena background, epic scale",
                "width": 192, "height": 128,
                "max_commands": 1000,
            },
            {
                "task_name": "complete_tileset_preview",
                "description": (
                    "a game-ready tileset preview sheet: "
                    "4x4 grid of matching tiles — grass, dirt, stone, water, sand, lava, ice, cloud, "
                    "all sharing a cohesive palette, seamless edges between adjacent tiles, "
                    "each tile 32x32, arranged neatly on a dark background, "
                    "polished to commercial game release standard"
                ),
                "style": "complete pixel art tileset sheet, game-ready",
                "width": 192, "height": 128,
                "max_commands": 1000,
            },
        ],
    },

    # ── LEVEL 7: OUTLINE MASTERY & CLEAN LINEWORK ───────────────────────────
    {
        "level": 7,
        "name": "Outline Mastery & Clean Linework",
        "description": (
            "Master the art of crisp, intentional outlines. Selective outlines (inner/outer), "
            "anti-pillow-shading, sub-pixel detail, shape-defining contour lines. "
            "Every stroke must carry meaning. This is what separates amateur pixel art from pro."
        ),
        "min_score": 91,
        "graduate_at": 96,
        "canvas_size": (128, 128),
        "max_commands": 1200,
        "tasks": [
            {
                "task_name": "selective_outline_character",
                "description": (
                    "an advanced pixel art warrior character with SELECTIVE OUTLINES: "
                    "outer silhouette = 1px dark outline (darkest shade of each material color, NOT black), "
                    "inner detail lines = thinner, lighter shade outlines, "
                    "NO outline between adjacent surfaces of same material, "
                    "outline weight varies — thicker at ground, thinner at top, "
                    "result: character looks carved from light, highly readable silhouette, "
                    "Owlboy / Metal Slug quality linework"
                ),
                "style": "selective outline pixel art, advanced linework mastery",
                "width": 80, "height": 128,
                "max_commands": 900,
            },
            {
                "task_name": "clean_outline_dragon",
                "description": (
                    "a GOTY-quality pixel art dragon head portrait: "
                    "massive horned dragon face — scales, ridges, glowing eyes, "
                    "CLEAN single-pixel outlines with NO jagged corners (use corner anti-aliasing), "
                    "every curved scale individually outlined, "
                    "inner wing membrane outlined differently from scales, "
                    "selective shading outlines inside each scale cluster, "
                    "reference: Blasphemous, Dragon's Crown pixel art quality"
                ),
                "style": "clean outline pixel art dragon, showcase-quality linework",
                "width": 128, "height": 128,
                "max_commands": 1200,
            },
            {
                "task_name": "architecture_outlines",
                "description": (
                    "a pixel art fantasy tower building: "
                    "clean architectural outlines — vertical walls have strong 1px edges, "
                    "window arches use curved 1px outlines with corner pixels removed, "
                    "stone blocks separated by 1px mortar line in darker stone shade, "
                    "NO pure black outlines — each material outlines in its own darkest color, "
                    "rooftop tiles have individual outline per tile, "
                    "Tactics Ogre / Final Fantasy Tactics building quality"
                ),
                "style": "architectural pixel art, professional game asset quality",
                "width": 96, "height": 128,
                "max_commands": 1000,
            },
            {
                "task_name": "multi_material_item",
                "description": (
                    "a pixel art legendary sword weapon: "
                    "blade (silver metal), guard (gold), grip (dark leather wrap), pommel (gemstone), "
                    "EACH material has its own outline color scheme: "
                    "blade outlined in steel-blue, gold outlined in dark amber, leather in dark brown, gem in deep purple, "
                    "magical runes etched INTO the blade with lighter pixel lines, "
                    "light source from top-right: specular highlight streak on blade, "
                    "Castlevania / Dark Souls weapon icon quality"
                ),
                "style": "multi-material pixel art weapon, selective outline mastery",
                "width": 64, "height": 128,
                "max_commands": 900,
            },
        ],
    },

    # ── LEVEL 8: 3D ILLUSION & EXTREME SHADING ──────────────────────────────
    {
        "level": 8,
        "name": "3D Illusion & Extreme Shading",
        "description": (
            "Make flat pixel art look fully 3D. Extreme 5-tone shading, rim lighting, "
            "ambient occlusion in pixel art, subsurface scattering suggestion, "
            "specular highlights, normal-map-style shading direction. "
            "The goal: someone should question if it's 3D rendered."
        ),
        "min_score": 93,
        "graduate_at": 97,
        "canvas_size": (192, 192),
        "max_commands": 1500,
        "tasks": [
            {
                "task_name": "sphere_3d_study",
                "description": (
                    "a pixel art sphere that looks photorealistic 3D: "
                    "5-tone shading gradient from light source (top-left): "
                    "  1=specular highlight (near-white), "
                    "  2=light face (bright base color), "
                    "  3=mid tone (base color), "
                    "  4=shadow (darker, slightly cooler hue), "
                    "  5=darkest shadow + ambient occlusion at bottom, "
                    "PLUS rim light on opposite edge from light source (cool blue tint), "
                    "cast shadow on ground below with soft pixel dithering at edge, "
                    "result must look like a CGI render, not like pixel art"
                ),
                "style": "3D-illusion pixel art, photorealistic shading study",
                "width": 128, "height": 128,
                "max_commands": 800,
            },
            {
                "task_name": "face_3d_portrait",
                "description": (
                    "a pixel art anime character face that looks 3D sculpted: "
                    "5-tone skin shading — forehead highlight, cheek mid, jaw shadow, neck ambient occlusion, "
                    "nose bridge catches light on one side, dark shadow on other, "
                    "lips: upper lip darker (in shadow), lower lip has specular highlight, "
                    "eyes: white + iris gradient + dark pupil + tiny white specular dot, "
                    "hair: clumps of strands each with highlight/shadow, "
                    "ears: inner ear ambient occlusion, outer edge rim lit, "
                    "reference: Arc System Works (Guilty Gear Strive) anime portrait quality"
                ),
                "style": "3D-illusion pixel art portrait, anime cel-shaded but volumetric",
                "width": 128, "height": 160,
                "max_commands": 1400,
            },
            {
                "task_name": "armor_3d_study",
                "description": (
                    "a pixel art knight's breastplate that looks like polished 3D metal: "
                    "curved pauldrons with reflective metal surface: "
                    "5 shading tones from dark navy to silver-white, "
                    "specular 'streak' highlights following the surface curvature, "
                    "ambient occlusion in all joints and crevices (dark 1-2px gaps), "
                    "rim lighting on all armor edges (slightly warm orange to suggest torchlight), "
                    "engravings cut INTO the metal with shadow on one side and highlight on other, "
                    "reference: Monster Hunter World concept art translated to pixel art"
                ),
                "style": "3D volumetric pixel art armor, metal material study",
                "width": 160, "height": 192,
                "max_commands": 1500,
            },
            {
                "task_name": "landscape_atmospheric_depth",
                "description": (
                    "a pixel art mountain range landscape with EXTREME atmospheric depth illusion: "
                    "8 depth layers using aerial perspective technique: "
                    "  layer 1 (far background): mountains at 10% opacity, heavily desaturated blue-grey, "
                    "  layers 2-4: progressively more saturated, slightly warmer, more contrast each layer, "
                    "  layers 5-6: midground trees/hills, full saturation, normal contrast, "
                    "  layer 7: foreground rocks/grass at full detail + saturation + darkest values, "
                    "  layer 8: extreme foreground blur suggestion (single-pixel oversized details), "
                    "God rays from sun through clouds using dithered semi-transparent pixels, "
                    "reference: Octopath Traveler / Blasphemous environmental depth"
                ),
                "style": "atmospheric perspective pixel art landscape, extreme depth illusion",
                "width": 192, "height": 128,
                "max_commands": 1500,
            },
        ],
    },

    # ── LEVEL 9: COLOR BLENDING & SHAPE MASTERY ─────────────────────────────
    {
        "level": 9,
        "name": "Color Blending & Shape Mastery",
        "description": (
            "Advanced color theory in pixel art: hue-shifting shadows, saturation gradients, "
            "dithering as color mixing, color temperature drama, non-standard organic shapes, "
            "shape-within-shape design, silhouette uniqueness. "
            "Design vocabulary that makes players instantly recognize YOUR art style."
        ),
        "min_score": 95,
        "graduate_at": 98,
        "canvas_size": (192, 192),
        "max_commands": 1800,
        "tasks": [
            {
                "task_name": "hue_shift_shading",
                "description": (
                    "a pixel art fire mage character demonstrating ADVANCED HUE-SHIFT SHADING: "
                    "skin: base=peach(10,290,80%), shadows SHIFT to purple-red NOT just darker, highlights SHIFT toward yellow NOT just lighter, "
                    "robes: base=deep blue, shadows shift to violet-black, highlights shift to cyan-white, "
                    "fire effects: base=orange, outer flames shift to deep red, inner core shifts to yellow-white, "
                    "hair: dark brown base, shadow shifts to near-black with slight cool blue, highlight shifts to warm amber, "
                    "NO grey shading anywhere — every shadow is a shifted hue, "
                    "reference: Lospec palette tutorials, Slynyrd hue-shift technique"
                ),
                "style": "hue-shift pixel art, advanced color theory demonstration",
                "width": 96, "height": 160,
                "max_commands": 1400,
            },
            {
                "task_name": "dithering_color_blend",
                "description": (
                    "a pixel art sunset sky landscape showcasing DITHERING AS COLOR BLENDING: "
                    "gradient sky using ONLY 8 colors but appearing as smooth 32-color gradient via dithering patterns, "
                    "use checkerboard dithering to blend adjacent colors at horizon, "
                    "use ordered dithering (Bayer matrix) for clouds, "
                    "use random/noise dithering for distant fog layer, "
                    "sun halo: pure yellow core → dithered orange → dithered red-orange → deep purple horizon, "
                    "silhouette foreground trees in pure black (maximum contrast), "
                    "reference: PICO-8 palette art, NES dithering masters"
                ),
                "style": "dithering technique showcase, color blending via patterns",
                "width": 192, "height": 128,
                "max_commands": 1500,
            },
            {
                "task_name": "organic_shape_creature",
                "description": (
                    "a pixel art creature/monster with RADICALLY UNIQUE ORGANIC SHAPES: "
                    "NOT humanoid — something completely alien: "
                    "body built from overlapping organic blob shapes, no straight lines, "
                    "limbs that branch and taper in unexpected directions, "
                    "surface covered in different-sized circular pustules/scales of varied shapes, "
                    "bioluminescent patches using pure saturated color against desaturated body, "
                    "multiple eyes of different sizes arranged asymmetrically, "
                    "shape silhouette must be instantly recognizable from pure black shadow, "
                    "reference: Hollow Knight's creature design philosophy"
                ),
                "style": "organic shape design, creature silhouette uniqueness study",
                "width": 192, "height": 192,
                "max_commands": 1800,
            },
            {
                "task_name": "color_temperature_drama",
                "description": (
                    "a pixel art scene with EXTREME COLOR TEMPERATURE CONTRAST: "
                    "left half lit by magical cold BLUE light (ice spell area), "
                    "right half lit by roaring WARM ORANGE fire, "
                    "the center where they meet: purple/magenta transition zone using dithering, "
                    "a character standing in center: left side of body is cold-lit (blue highlights, purple mid), "
                    "right side is warm-lit (orange highlights, amber mid), "
                    "ground reflections mirror the color split, "
                    "shadows in cold area use warm orange complement, shadows in warm area use cool blue complement, "
                    "reference: Hades color lighting design, Disco Elysium color drama"
                ),
                "style": "color temperature contrast, dramatic pixel art lighting study",
                "width": 192, "height": 160,
                "max_commands": 1600,
            },
        ],
    },

    # ── LEVEL 10: LEGENDARY — EXTREME LAYERING & SIGNATURE STYLE ────────────
    {
        "level": 10,
        "name": "LEGENDARY — Extreme Layering & Signature Style",
        "description": (
            "The final form. 10+ depth layers, parallax-ready assets, "
            "a completely unique design language that no other game has. "
            "Ultra-dense detail that rewards zooming in. Cinematic composition. "
            "This is the level that wins Game of the Year for art direction alone. "
            "Training NEVER STOPS here — each episode pushes harder than the last."
        ),
        "min_score": 97,
        "graduate_at": 135,  # DIVINE — Redefines Gaming History (theoretical max)
        "canvas_size": (256, 192),
        "max_commands": 2500,
        "tasks": [
            {
                "task_name": "10layer_world_scene",
                "description": (
                    "the ULTIMATE pixel art world scene with 10 DISTINCT DEPTH LAYERS: "
                    "Layer 1 (sky): gradient sky with stars, distant galaxy smear, "
                    "Layer 2 (far mountains): silhouette mountains 5% detail, heavily aerial-perspective desaturated, "
                    "Layer 3 (floating islands): distant magical islands with tiny waterfalls, "
                    "Layer 4 (castle): mid-distance fantasy castle with tiny lit windows, "
                    "Layer 5 (tree canopy far): treetop silhouettes, mostly shape no detail, "
                    "Layer 6 (midground road): cobblestone road with worn center path, "
                    "Layer 7 (vegetation mid): bushes, tall grass, wildflowers, "
                    "Layer 8 (foreground rocks): large detailed boulders with moss and lichen, "
                    "Layer 9 (foreground vegetation): oversized grass blades, flowers with individual petal detail, "
                    "Layer 10 (extreme foreground): blurred bokeh-style large colored circles (out of focus), "
                    "EACH LAYER must be separable for Godot parallax implementation, "
                    "reference: Owlboy, Blasphemous, Hollow Knight background quality"
                ),
                "style": "10-layer parallax pixel art, LEGENDARY quality, Godot-ready",
                "width": 256, "height": 192,
                "max_commands": 2500,
            },
            {
                "task_name": "signature_style_character",
                "description": (
                    "create a pixel art character with a COMPLETELY UNIQUE SIGNATURE STYLE: "
                    "invent a new art style that has NEVER been seen — combine: "
                    "anime proportions (large eyes, expressive) + "
                    "Western comic book inking (thick outlines, crosshatch shadow) + "
                    "traditional Japanese woodblock print color palette (limited, high contrast) + "
                    "modern neon cyberpunk accent lights (hot pink, electric blue) + "
                    "the character itself must look like it belongs in NO existing game, "
                    "unusual costume shapes — no standard fantasy tropes, "
                    "a silhouette you have never seen in any other game, "
                    "if someone sees this character, they immediately know what game it's from"
                ),
                "style": "signature unique pixel art style, unprecedented design language",
                "width": 160, "height": 192,
                "max_commands": 2000,
            },
            {
                "task_name": "extreme_detail_environment",
                "description": (
                    "a pixel art environment with EXTREME MICRO-DETAIL that rewards zooming in: "
                    "an ancient library interior at isometric angle: "
                    "MACRO level: grand arched ceiling, massive bookshelves, floating magical orbs, "
                    "MESO level: individual books with different spine colors and thicknesses, "
                    "MICRO level (zoom in): tiny readable letters on book spines, "
                    "cobwebs in ceiling corners with individual web strand pixels, "
                    "dust particles caught in light shafts (single pixel dots), "
                    "individual wood grain lines on floorboards, "
                    "tiny mouse peering from a hole in the baseboard, "
                    "EVERY 16x16 region of the canvas must contain unique detail, "
                    "reference: Stardew Valley farm building interiors, but 10x more detailed"
                ),
                "style": "extreme micro-detail pixel art environment, multi-scale detail density",
                "width": 256, "height": 192,
                "max_commands": 2500,
            },
            {
                "task_name": "shape_language_showcase",
                "description": (
                    "a pixel art SHAPE LANGUAGE SHOWCASE for Isekai Chronicles: "
                    "4 character archetypes each defined by a dominant geometric shape: "
                    "HERO (triangular — sharp angles, forward-leaning posture, pointed armor), "
                    "MAGE (circular — robes flowing in circles, round hat, orb motifs), "
                    "VILLAIN (angular jagged — asymmetric, sharp spikes, broken shapes), "
                    "HEALER (soft rectangular — wide, stable, comforting proportions), "
                    "arranged in a 2x2 grid, each 64x96px, "
                    "shape language must be INSTANTLY readable without color, "
                    "use ONLY black silhouette to tell each character's personality, "
                    "reference: Pixar's shape language theory applied to pixel art"
                ),
                "style": "shape language design study, character archetype silhouettes",
                "width": 256, "height": 192,
                "max_commands": 2000,
            },
            {
                "task_name": "multi_style_fusion",
                "description": (
                    "the ULTIMATE FUSION PIECE for Isekai Chronicles: "
                    "a single pixel art image that fuses ALL skills: "
                    "SUBJECT: the game's hero facing the final boss dragon in an epic confrontation, "
                    "OUTLINE: selective outlines, no black, each material its own outline color, "
                    "SHADING: 5-tone extreme shading making everything look 3D sculpted, "
                    "COLOR: hue-shifted shadows, complementary rim lights, dithered gradients, "
                    "LAYERS: 8+ depth planes with atmospheric perspective, "
                    "SHAPES: hero=triangle energy, dragon=circular coiling mass + jagged teeth, "
                    "DETAIL: micro details reward zooming — individual dragon scales, hero sweat drops, "
                    "COMPOSITION: rule of thirds, hero bottom-left, dragon fills top-right, "
                    "lighting drama from dragon's fire breath illuminating hero from below-right, "
                    "this image must make anyone who sees it immediately want to play the game"
                ),
                "style": "LEGENDARY fusion pixel art, all techniques combined, game cover quality",
                "width": 256, "height": 192,
                "max_commands": 2500,
            },
        ],
    },
]


def get_current_level(rolling_avg: float) -> Dict[str, Any]:
    """Get the appropriate curriculum level for the current skill level."""
    for level in reversed(DRAWING_CURRICULUM):
        if rolling_avg >= level["min_score"]:
            return level
    return DRAWING_CURRICULUM[0]


def get_next_task(level: Dict, episode: int) -> Dict[str, Any]:
    """Cycle through tasks in the current level."""
    tasks = level["tasks"]
    task = dict(tasks[episode % len(tasks)])
    return task


def get_level_count() -> int:
    return len(DRAWING_CURRICULUM)
