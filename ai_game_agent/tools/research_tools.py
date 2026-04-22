"""
Research tool — searches YouTube, AniList, and the web for:
  • Isekai/anime inspiration (class systems, magic, world design)
  • Game design references (mechanics, maps, UI patterns)
  • Pixel art tutorials and style guides

Results are summarised and fed back as LLM context.
"""
from __future__ import annotations
import json, re, time
from typing import Optional
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AIGameAgent/0.1)"}

# ──────────────────────────────────────────────────────────────────────────────
# YouTube search (no API key needed — uses web scrape of search results)
# ──────────────────────────────────────────────────────────────────────────────

def search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """Return a list of YouTube video metadata for a search query."""
    url = f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        # Extract ytInitialData JSON embedded in the page
        match = re.search(r'var ytInitialData = (\{.*?\});', r.text, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group(1))
        contents = (
            data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
        )
        videos = []
        for section in contents:
            items = section.get("itemSectionRenderer", {}).get("contents", [])
            for item in items:
                vr = item.get("videoRenderer")
                if not vr:
                    continue
                vid_id = vr.get("videoId", "")
                title = "".join(
                    r.get("text", "") for r in
                    vr.get("title", {}).get("runs", [])
                )
                desc = "".join(
                    r.get("text", "") for r in
                    vr.get("descriptionSnippet", {}).get("runs", [])
                ) if "descriptionSnippet" in vr else ""
                channel = (
                    vr.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
                )
                videos.append({
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                    "channel": channel,
                    "description": desc,
                })
                if len(videos) >= max_results:
                    break
            if len(videos) >= max_results:
                break
        return videos
    except Exception as e:
        return [{"error": str(e)}]


# ──────────────────────────────────────────────────────────────────────────────
# AniList GraphQL — top isekai anime with descriptions
# ──────────────────────────────────────────────────────────────────────────────

ANILIST_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($page: Int, $perPage: Int, $genre: String) {
  Page(page: $page, perPage: $perPage) {
    media(genre: $genre, type: ANIME, sort: POPULARITY_DESC) {
      title { english romaji }
      genres
      description(asHtml: false)
      averageScore
      episodes
      tags { name rank }
    }
  }
}
"""

def fetch_isekai_anime(page: int = 1, per_page: int = 10) -> list[dict]:
    """Fetch popular isekai anime with descriptions from AniList."""
    try:
        r = requests.post(
            ANILIST_URL,
            json={"query": ANILIST_QUERY, "variables": {"page": page, "perPage": per_page, "genre": "Isekai"}},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        media_list = r.json()["data"]["Page"]["media"]
        results = []
        for m in media_list:
            title = m["title"].get("english") or m["title"].get("romaji", "Unknown")
            desc = re.sub(r"<.*?>", "", m.get("description") or "")[:400]
            top_tags = [t["name"] for t in sorted(m.get("tags", []), key=lambda t: -t["rank"])[:6]]
            results.append({
                "title": title,
                "score": m.get("averageScore"),
                "episodes": m.get("episodes"),
                "genres": m.get("genres", []),
                "tags": top_tags,
                "description": desc,
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ──────────────────────────────────────────────────────────────────────────────
# DuckDuckGo text search (no API key)
# ──────────────────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via DuckDuckGo HTML and return snippets."""
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for res in soup.select(".result__body")[:max_results]:
            title_el = res.select_one(".result__title")
            snippet_el = res.select_one(".result__snippet")
            link_el = res.select_one(".result__url")
            results.append({
                "title": title_el.get_text(strip=True) if title_el else "",
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                "url": link_el.get_text(strip=True) if link_el else "",
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ──────────────────────────────────────────────────────────────────────────────
# High-level research functions
# ──────────────────────────────────────────────────────────────────────────────

def research_isekai_ideas(topic: str = "isekai RPG game design") -> dict:
    """
    Comprehensive research pass: YouTube tutorials + AniList top isekai + web search.
    Returns a structured dict ready to be injected as LLM context.
    """
    print(f"[Research] Searching YouTube: '{topic}'...")
    yt = search_youtube(f"{topic} pixel art RPG game", max_results=5)

    print("[Research] Fetching top isekai anime from AniList...")
    anime = fetch_isekai_anime(per_page=8)

    print(f"[Research] Web search: '{topic} mechanics'...")
    web = web_search(f"{topic} game mechanics worldbuilding", max_results=5)

    return {
        "youtube_videos": yt,
        "isekai_anime": anime,
        "web_results": web,
    }


def research_pixel_art_style(style: str = "pixel art RPG top-down") -> dict:
    """Search for pixel art tutorials and style references."""
    yt = search_youtube(f"{style} tutorial Godot", max_results=5)
    web = web_search(f"{style} tileset design guide", max_results=5)
    return {"youtube_videos": yt, "web_results": web}


def research_game_mechanics(mechanic: str) -> dict:
    """Research a specific game mechanic (e.g. 'class system', 'village simulation')."""
    yt = search_youtube(f"{mechanic} game design RPG", max_results=4)
    web = web_search(f"{mechanic} game design best practices pixel RPG", max_results=4)
    return {"youtube_videos": yt, "web_results": web}


def summarise_research_for_llm(research: dict) -> str:
    """Convert research dict into a concise text block for LLM context."""
    lines = ["=== RESEARCH CONTEXT ===\n"]

    if research.get("isekai_anime"):
        lines.append("## Top Isekai Anime (AniList)\n")
        for a in research["isekai_anime"][:5]:
            if "error" in a:
                continue
            lines.append(f"• {a['title']} (Score: {a.get('score','?')}/100)")
            lines.append(f"  Tags: {', '.join(a.get('tags', []))}")
            if a.get("description"):
                lines.append(f"  Synopsis: {a['description'][:200]}...")
            lines.append("")

    if research.get("youtube_videos"):
        lines.append("## YouTube References\n")
        for v in research["youtube_videos"][:4]:
            if "error" in v:
                continue
            lines.append(f"• [{v.get('title','')}] by {v.get('channel','')}")
            if v.get("description"):
                lines.append(f"  {v['description'][:150]}")
            lines.append(f"  {v.get('url','')}")
        lines.append("")

    if research.get("web_results"):
        lines.append("## Web References\n")
        for w in research["web_results"][:4]:
            if "error" in w:
                continue
            lines.append(f"• {w.get('title','')}")
            lines.append(f"  {w.get('snippet','')[:200]}")
        lines.append("")

    return "\n".join(lines)
