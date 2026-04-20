"""Tile context to LLM prompt assembly."""

from enum import Enum

class FormatStyle(Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"
    PLAIN = "plain"

class TilePromptBuilder:
    def __init__(self, max_budget: int = 2000, format_style: FormatStyle = FormatStyle.MARKDOWN):
        self.max_budget = max_budget
        self.format_style = format_style

    def format_tile(self, tile: dict) -> str:
        content = tile.get("content", "")
        conf = tile.get("confidence", 0.5)
        domain = tile.get("domain", "")
        if self.format_style == FormatStyle.MARKDOWN:
            return f"- [{domain}] {content} (confidence: {conf:.2f})"
        elif self.format_style == FormatStyle.JSON:
            return f'{{"domain":"{domain}","content":"{content}","confidence":{conf}}}'
        elif self.format_style == FormatStyle.XML:
            return f'<tile domain="{domain}" confidence="{conf}">{content}</tile>'
        else:
            return content

    def build(self, tiles: list[dict], query: str = "", system: str = "") -> str:
        sorted_tiles = sorted(tiles, key=lambda t: t.get("confidence", 0), reverse=True)
        parts = []
        budget = self.max_budget
        if system:
            parts.append(f"System: {system}")
            budget -= len(system) + 8
        if query:
            parts.append(f"Query: {query}")
            budget -= len(query) + 7
        for t in sorted_tiles:
            formatted = self.format_tile(t)
            if len(formatted) > budget:
                continue
            parts.append(formatted)
            budget -= len(formatted)
        return "\n".join(parts)

    def build_with_deadband(self, tiles: list[dict], query: str = "",
                            system: str = "", excluded_p0: list[dict] = None) -> str:
        prompt = self.build(tiles, query, system)
        if excluded_p0:
            notice = f"\n[!P0 tiles excluded by budget: {len(excluded_p0)}]"
            prompt += notice
        return prompt

    def budget_remaining(self, tiles: list[dict], query: str = "", system: str = "") -> int:
        used = len(query) + len(system)
        for t in tiles:
            used += len(self.format_tile(t))
        return max(self.max_budget - used, 0)

    @property
    def stats(self) -> dict:
        return {"max_budget": self.max_budget, "format": self.format_style.value}
