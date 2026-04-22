"""
GOTY World Evaluator
====================
Scores AI Dev's game world designs against a Game of the Year rubric.

Benchmark games studied:
  - Stardew Valley   (world cohesion, NPC depth, progression)
  - Hades            (game feel, narrative integration, replayability)
  - Hollow Knight    (atmosphere, exploration, visual storytelling)
  - CrossCode        (systems depth, pacing, technical polish)
  - Undertale        (emotional resonance, originality, player agency)

10 Scoring Dimensions (10 pts each = 100 base):
  1.  World Cohesion     — biomes/areas feel connected, make narrative sense
  2.  Narrative Depth    — story told through environment + characters
  3.  Player Agency      — meaningful choices, multiple paths, player expression
  4.  Game Feel          — input response, juice, feedback, camera, transitions
  5.  NPC / AI Depth     — schedules, memory, relationships, emergent behavior
  6.  Systems Depth      — crafting, economy, progression, magic, status effects
  7.  Visual Storytelling — art communicates world lore without text
  8.  Technical Polish   — no crashes, 60fps, clean UI, save/load, menus
  9.  Replayability      — procedural elements, branching, secrets, NG+
  10. Emotional Hook     — memorable moments, music cues, character arcs

LEGENDARY BONUS (can push score beyond 100):
  Up to +15 bonus points for innovations that no GOTY has fully achieved:
  • AI-Driven NPC Memory (+5)   — NPCs remember and react to past player actions
  • Procedural Narrative (+5)   — story generates dynamically from player choices
  • Living Ecosystem (+3)       — world evolves/grows when player isn't watching
  • Cross-Biome Physics (+2)    — weather/seasons affect all systems coherently

Usage:
    evaluator = GOTYEvaluator()
    result = evaluator.evaluate(world_description="...", code_files={...})
    print(result.report())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── GOTY Benchmark Rubric ────────────────────────────────────────────────────

# What separates GOTY games from merely good ones
GOTY_STANDARDS = {
    "stardew_valley": {
        "world_cohesion": 9.5,
        "narrative_depth": 8.5,
        "player_agency": 9.0,
        "game_feel": 8.5,
        "npc_depth": 9.5,      # 28 fully voiced villagers with schedules
        "systems_depth": 9.5,  # farming, mining, fishing, combat, crafting
        "visual_storytelling": 8.0,
        "technical_polish": 9.0,
        "replayability": 9.0,
        "emotional_hook": 9.5,
    },
    "hades": {
        "world_cohesion": 9.5,
        "narrative_depth": 10.0,  # story advances every run
        "player_agency": 9.5,
        "game_feel": 10.0,        # industry benchmark for game feel
        "npc_depth": 9.5,
        "systems_depth": 9.0,
        "visual_storytelling": 9.5,
        "technical_polish": 9.5,
        "replayability": 10.0,
        "emotional_hook": 9.5,
    },
    "hollow_knight": {
        "world_cohesion": 10.0,
        "narrative_depth": 9.0,
        "player_agency": 8.5,
        "game_feel": 9.5,
        "npc_depth": 8.5,
        "systems_depth": 8.0,
        "visual_storytelling": 10.0,
        "technical_polish": 9.0,
        "replayability": 8.5,
        "emotional_hook": 9.5,
    },
}

# Average GOTY benchmark score per dimension
GOTY_BENCHMARK: dict[str, float] = {
    dim: sum(game[dim] for game in GOTY_STANDARDS.values()) / len(GOTY_STANDARDS)
    for dim in list(GOTY_STANDARDS.values())[0]
}


# ── Score result ─────────────────────────────────────────────────────────────

@dataclass
class GOTYResult:
    scores: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    grade: str = ""
    gaps: list[str] = field(default_factory=list)      # what's missing vs GOTY
    strengths: list[str] = field(default_factory=list)  # what's already GOTY-level
    next_targets: list[str] = field(default_factory=list)  # top 3 things to improve
    legendary_bonuses: dict[str, float] = field(default_factory=dict)  # name → bonus pts

    def report(self) -> str:
        base_score = self.total - sum(self.legendary_bonuses.values())
        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            f"  🏆 GOTY EVALUATION — {self.total:.1f}/100  |  {self.grade}",
            "╠══════════════════════════════════════════════════════════╣",
        ]
        dim_names = {
            "world_cohesion":     "World Cohesion    ",
            "narrative_depth":    "Narrative Depth   ",
            "player_agency":      "Player Agency     ",
            "game_feel":          "Game Feel         ",
            "npc_depth":          "NPC / AI Depth    ",
            "systems_depth":      "Systems Depth     ",
            "visual_storytelling":"Visual Storytelling",
            "technical_polish":   "Technical Polish  ",
            "replayability":      "Replayability     ",
            "emotional_hook":     "Emotional Hook    ",
        }
        for key, label in dim_names.items():
            score = self.scores.get(key, 0.0)
            bench = GOTY_BENCHMARK.get(key, 9.0)
            bar = "█" * int(score) + "░" * (10 - int(score))
            gap_marker = " ✅" if score >= bench * 0.9 else f" ←{bench*10:.0f} GOTY"
            lines.append(f"  {label}: {bar} {score*10:.0f}/100{gap_marker}")
        lines.append(f"  {'─'*54}")
        lines.append(f"  Base score  : {base_score:.1f}/100")
        if self.legendary_bonuses:
            bonus_total = sum(self.legendary_bonuses.values())
            lines.append(f"  🌌 LEGENDARY BONUSES: +{bonus_total:.0f} pts")
            for name, pts in self.legendary_bonuses.items():
                lines.append(f"      {name}: +{pts:.0f}")
            lines.append(f"  FINAL SCORE : {self.total:.1f}/100  ← BEYOND GOTY!")
        lines.append("╠══════════════════════════════════════════════════════════╣")
        if self.strengths:
            lines.append("  💪 Strengths: " + " | ".join(self.strengths[:3]))
        if self.next_targets:
            lines.append("  🎯 Top improvements needed:")
            for t in self.next_targets[:3]:
                lines.append(f"      → {t}")
        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# ── Individual Dimension Evaluators ─────────────────────────────────────────

class WorldCohesionEvaluator:
    """Does the world feel like a real, connected place?"""

    STRONG = [
        (r"biome|region|zone|district|quarter", 2, "Has distinct areas"),
        (r"border|transition|connect|bridge|road|path", 2, "Areas are connected"),
        (r"capital|village|city|town|settlement|hub", 2, "Has population centers"),
        (r"wilderness|dungeon|ruin|cave|forest|mountain", 2, "Has wild/danger areas"),
        (r"history|lore|legend|ancient|relic|past", 2, "Has worldbuilding history"),
    ]
    WEAK = [
        (r"random\s+map|procedural.*only|no.*story", -3, "Random with no narrative"),
        (r"generic|placeholder|TODO.*world|example.*map", -4, "Placeholder design"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class NarrativeDepthEvaluator:
    """Story told through world + characters, not just cutscenes."""

    STRONG = [
        (r"quest|mission|storyline|arc|chapter", 2, "Has quest structure"),
        (r"backstory|origin|motivation|goal|dream|fear", 2, "Characters have depth"),
        (r"conflict|war|betrayal|mystery|secret|prophecy", 2, "Has narrative conflict"),
        (r"dialogue|conversation|journal|letter|book|note", 2, "Has environmental narrative"),
        (r"consequence|choice.*matter|branch|ending", 2, "Choices have weight"),
    ]
    WEAK = [
        (r"kill.*all|no.*story|kill.*enemies|survive.*only", -3, "No narrative beyond killing"),
        (r"placeholder.*story|TODO.*quest|story.*later", -4, "Story deferred"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class PlayerAgencyEvaluator:
    """Meaningful choices, expression, and multiple paths."""

    STRONG = [
        (r"class|job|build|spec|archetype|playstyle", 2, "Character customization"),
        (r"branch|choice|decision|vote|side.*with|faction", 2, "Branching decisions"),
        (r"optional|secret|hidden|discover|explore", 2, "Optional exploration"),
        (r"moral|alignment|karma|reputation|relationship", 2, "Moral dimensions"),
        (r"craft|build|design|customize|modify|upgrade", 2, "Player expression"),
    ]
    WEAK = [
        (r"linear.*only|no.*choice|one.*path|forced.*route", -3, "Linear with no agency"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class GameFeelEvaluator:
    """Input response, juice, feedback loops — the Hades standard."""

    STRONG = [
        (r"screen.?shake|camera.?shake|impact|hit.?stop|hitstop", 2, "Impact feedback"),
        (r"particle|vfx|effect|sparkle|dust|blood|slash", 2, "Visual feedback"),
        (r"sound.?effect|sfx|audio|cue|feedback.*sound", 2, "Audio feedback"),
        (r"smooth|lerp|tween|ease|interpolat", 2, "Smooth animations/transitions"),
        (r"responsive|instant|frame.*perfect|coyote|buffer", 2, "Responsive controls"),
    ]
    WEAK = [
        (r"no.*sound|no.*effect|no.*animation|placeholder.*art", -2, "Missing feel elements"),
        (r"lag|stutter|freeze|delay.*input", -3, "Input lag/stutter"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class NPCDepthEvaluator:
    """NPCs with schedules, memory, relationships — the Stardew standard."""

    STRONG = [
        (r"schedule|routine|daily|morning|night|sleep|work", 2, "NPC daily schedules"),
        (r"remember|memory|relationship|friendship|love|hate|trust", 2, "NPC memory/relationships"),
        (r"dialogue.*tree|conversation.*branch|respond.*to|react.*to", 2, "Contextual NPC dialogue"),
        (r"faction|guild|family|tribe|clan|allegiance", 2, "Social structures"),
        (r"goap|behavior.*tree|blackboard|ai.*state|npc.*goal", 2, "Advanced NPC AI"),
    ]
    WEAK = [
        (r"static.*npc|no.*dialogue|enemy.*only|dumb.*ai", -3, "No NPC depth"),
        (r"npc.*TODO|placeholder.*npc|add.*later.*npc", -4, "NPCs deferred"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class SystemsDepthEvaluator:
    """Interlocking game systems — the CrossCode/Stardew standard."""

    STRONG = [
        (r"craft|recipe|ingredient|combine|forge|brew", 2, "Crafting system"),
        (r"economy|trade|merchant|gold|currency|market|price", 2, "Economy system"),
        (r"level.*up|xp|experience|skill.*tree|talent|perk", 2, "Progression system"),
        (r"magic|spell|mana|ability|skill|power|element", 2, "Magic/ability system"),
        (r"status.*effect|buff|debuff|poison|burn|freeze|bleed", 2, "Status effects"),
    ]
    WEAK = [
        (r"one.*system.*only|no.*crafting|no.*progression|bare.*minimum", -3, "Shallow systems"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class VisualStorytellingEvaluator:
    """Art communicates story without words — the Hollow Knight standard."""

    STRONG = [
        (r"tileset|biome.*art|environment.*art|pixel.*art|sprite", 2, "Has custom art"),
        (r"lighting|shadow|glow|ambient|atmosphere|mood", 2, "Atmospheric lighting"),
        (r"ruin|decay|overgrown|broken|ancient.*visual", 2, "Environment tells history"),
        (r"parallax|layer|depth|background|foreground", 2, "Visual depth"),
        (r"color.*palette|consistent.*style|cohesive.*art|art.*direction", 2, "Cohesive art style"),
    ]
    WEAK = [
        (r"placeholder.*art|no.*art|programmer.*art|boxes.*only", -3, "No visual storytelling"),
        (r"inconsistent.*style|mixed.*art|clashing.*visual", -2, "Inconsistent visual style"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class TechnicalPolishEvaluator:
    """No crashes, 60fps, clean UI — production ready."""

    STRONG = [
        (r"save.*load|autosave|checkpoint|persist", 2, "Save/load system"),
        (r"menu|title.*screen|main.*menu|pause.*menu|options", 2, "Complete menus"),
        (r"loading.*screen|transition|scene.*change|fade", 2, "Scene transitions"),
        (r"settings|volume|resolution|fullscreen|accessibility", 2, "Settings/accessibility"),
        (r"60.*fps|performance|optimize|cull|lod|batch", 2, "Performance optimization"),
    ]
    WEAK = [
        (r"crash|freeze|bug.*known|TODO.*fix|broken.*feature", -3, "Known bugs/crashes"),
        (r"no.*menu|no.*save|no.*ui|no.*settings", -2, "Missing core polish"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class ReplayabilityEvaluator:
    """Reasons to play again — the Hades standard."""

    STRONG = [
        (r"procedural|random.*generat|roguelite|roguelike|permadeath", 2, "Procedural elements"),
        (r"new.*game.*plus|ng\+|hard.*mode|challenge|difficulty", 2, "NG+ / difficulty modes"),
        (r"secret|hidden.*area|easter.*egg|unlockable|collectible", 2, "Secrets to discover"),
        (r"multiple.*ending|branching.*story|choice.*affect", 2, "Multiple endings"),
        (r"achievement|trophy|challenge.*mode|speedrun|score.*attack", 2, "Extended goals"),
    ]
    WEAK = [
        (r"one.*ending.*only|linear.*no.*replay|no.*secret", -3, "Nothing new on replay"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


class EmotionalHookEvaluator:
    """Memorable moments, music, character arcs — what players remember."""

    STRONG = [
        (r"music|soundtrack|theme|leitmotif|ambient.*sound|score", 2, "Musical identity"),
        (r"memorable|iconic|twist|revelation|shock|beautiful|tragic", 2, "Memorable moments"),
        (r"character.*arc|growth|change|transform|redemption|loss", 2, "Character arcs"),
        (r"humor|comedy|joke|witty|banter|personality|charm", 2, "Personality/charm"),
        (r"dark|melancholy|hope|despair|triumph|sacrifice|love", 2, "Emotional range"),
    ]
    WEAK = [
        (r"no.*story|no.*character|bland|generic.*world|no.*music", -3, "Emotionally flat"),
    ]

    def score(self, text: str) -> tuple[float, list[str], list[str]]:
        s, good, bad = 0.0, [], []
        for pat, pts, label in self.STRONG:
            if re.search(pat, text, re.I): s += pts; good.append(label)
        for pat, pts, label in self.WEAK:
            if re.search(pat, text, re.I): s += pts; bad.append(label)
        return max(0.0, min(10.0, s)), good, bad


# ── Master GOTY Evaluator ────────────────────────────────────────────────────

DIMENSION_LABELS = {
    "world_cohesion":      "World Cohesion",
    "narrative_depth":     "Narrative Depth",
    "player_agency":       "Player Agency",
    "game_feel":           "Game Feel",
    "npc_depth":           "NPC/AI Depth",
    "systems_depth":       "Systems Depth",
    "visual_storytelling": "Visual Storytelling",
    "technical_polish":    "Technical Polish",
    "replayability":       "Replayability",
    "emotional_hook":      "Emotional Hook",
}

GRADE_LABELS = [
    (135, "👑 DIVINE — REDEFINES GAMING HISTORY"),
    (130, "🔱 MYTHIC — UNTOUCHABLE MASTERPIECE"),
    (125, "🌠 CELESTIAL — GENRE-DEFINING FOREVER"),
    (120, "💫 GODLIKE — BEYOND ALL COMPETITION"),
    (115, "🌌 LEGENDARY — BEYOND GOTY"),
    (110, "⚡ TRANSCENDENT"),
    (105, "✨ LEGENDARY CONTENDER"),
    (100, "🏆 PERFECT GOTY"),
    (95,  "🏆 GOTY CONTENDER"),
    (90,  "🌟 AAA RELEASE QUALITY"),
    (85,  "💎 CRITICALLY ACCLAIMED"),
    (80,  "🔥 GREAT GAME"),
    (75,  "⚡ GOOD GAME"),
    (65,  "✨ SOLID FOUNDATION"),
    (50,  "🎯 IN DEVELOPMENT"),
    (0,   "🌱 EARLY CONCEPT"),
]

# ── Legendary Bonus Dimensions ───────────────────────────────────────────────
# Features that NO GOTY has fully achieved — AI innovations beyond the rubric

LEGENDARY_BONUSES = [
    (
        r"npc.*remember|remember.*action|memory.*npc|relationship.*history"
        r"|npc.*react.*past|persistent.*memory|recall.*event",
        5.0,
        "🧠 AI-Driven NPC Memory",
        "NPCs remember and adapt to every player action across sessions",
    ),
    (
        r"procedural.*story|dynamic.*narrative|emergent.*narrative"
        r"|story.*generate|narrative.*evolve|generated.*plot",
        5.0,
        "📖 Procedural Narrative",
        "Story generates dynamically from player choices, no two playthroughs identical",
    ),
    (
        r"living.*world|world.*evolve|ecosystem.*grow|npc.*build"
        r"|world.*change.*offline|simulation.*offline|colony.*grow",
        3.0,
        "🌍 Living Ecosystem",
        "World evolves and grows autonomously even when the player isn't watching",
    ),
    (
        r"season.*affect|weather.*system|cross.*biome.*physics"
        r"|biome.*weather|season.*crop|dynamic.*weather.*gameplay",
        2.0,
        "🌦 Cross-Biome Physics",
        "Weather and seasons coherently affect all gameplay systems simultaneously",
    ),
]


class GOTYEvaluator:
    """
    Master GOTY evaluator. Pass any combination of:
      - world_description: text description of the game world
      - code_text: combined GDScript code
      - design_doc: design document or bullet-point notes
    """

    def __init__(self):
        self._evals = {
            "world_cohesion":      WorldCohesionEvaluator(),
            "narrative_depth":     NarrativeDepthEvaluator(),
            "player_agency":       PlayerAgencyEvaluator(),
            "game_feel":           GameFeelEvaluator(),
            "npc_depth":           NPCDepthEvaluator(),
            "systems_depth":       SystemsDepthEvaluator(),
            "visual_storytelling": VisualStorytellingEvaluator(),
            "technical_polish":    TechnicalPolishEvaluator(),
            "replayability":       ReplayabilityEvaluator(),
            "emotional_hook":      EmotionalHookEvaluator(),
        }

    def evaluate(
        self,
        world_description: str = "",
        code_text: str = "",
        design_doc: str = "",
    ) -> GOTYResult:
        # Combine all text for searching
        full_text = " ".join([world_description, code_text, design_doc]).lower()

        result = GOTYResult()
        all_strengths: list[str] = []
        all_gaps: list[tuple[str, float]] = []  # (label, gap_amount)

        for dim, evaluator in self._evals.items():
            score, goods, bads = evaluator.score(full_text)
            result.scores[dim] = score

            bench_score = GOTY_BENCHMARK[dim]
            if score >= bench_score * 0.9:
                all_strengths += [f"{DIMENSION_LABELS[dim]}: {score*10:.0f}/100"]
            elif score < bench_score * 0.6:
                gap = bench_score - score
                all_gaps.append((DIMENSION_LABELS[dim], gap))

            result.strengths += [f"{DIMENSION_LABELS[dim]}: {s}" for s in goods[:1]]

        # Base score (each dimension is 0-10, total is 0-100)
        result.total = sum(result.scores.values()) * 10 / len(result.scores)

        # ── Legendary Bonus — can push past 100 ──────────────────────────────
        result.legendary_bonuses = {}
        if result.total >= 90.0:  # Only unlocks when already GOTY-caliber
            for pattern, pts, name, desc in LEGENDARY_BONUSES:
                if re.search(pattern, full_text, re.I):
                    result.legendary_bonuses[name] = pts
                    result.total += pts

        # Grade (thresholds now go up to 115+)
        for threshold, label in GRADE_LABELS:
            if result.total >= threshold:
                result.grade = label
                break

        # Top 3 gaps to fix (sorted by severity)
        all_gaps.sort(key=lambda x: -x[1])
        result.next_targets = [
            f"Improve {label} (currently {result.scores.get(k, 0)*10:.0f}/100, "
            f"GOTY standard is {GOTY_BENCHMARK[k]*10:.0f}/100)"
            for label, gap in all_gaps[:3]
            for k, v in DIMENSION_LABELS.items() if v == label
        ]

        result.gaps = [label for label, _ in all_gaps]
        result.strengths = all_strengths[:5]

        return result

    def compare_to_benchmark(self, result: GOTYResult) -> str:
        """Show how AI Dev compares to GOTY benchmark games."""
        lines = ["\n📊 GOTY BENCHMARK COMPARISON", "-" * 50]
        for dim, label in DIMENSION_LABELS.items():
            score = result.scores.get(dim, 0.0) * 10
            bench = GOTY_BENCHMARK[dim] * 10
            diff = score - bench
            arrow = "↑" if diff >= 0 else "↓"
            lines.append(
                f"  {label:<22}: {score:5.0f}/100  {arrow} {abs(diff):4.0f}  "
                f"(GOTY avg: {bench:.0f})"
            )
        lines.append("-" * 50)
        base = result.total - sum(result.legendary_bonuses.values()) if result.legendary_bonuses else result.total
        lines.append(f"  AI Dev base  : {base:.1f}/100")
        if result.legendary_bonuses:
            bonus_total = sum(result.legendary_bonuses.values())
            lines.append(f"  Legendary    : +{bonus_total:.0f} pts")
            for name, pts in result.legendary_bonuses.items():
                lines.append(f"    {name}: +{pts:.0f}")
        lines.append(f"  FINAL SCORE  : {result.total:.1f}/100")
        lines.append(f"  GOTY average : {sum(GOTY_BENCHMARK.values()) * 10 / len(GOTY_BENCHMARK):.1f}/100")
        return "\n".join(lines)
