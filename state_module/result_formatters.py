"""
Result formatters for different tool output types.
Provides pluggable formatters to avoid hard-coding result parsing logic.
"""

import json
from typing import Any, Dict, Optional, Callable


class ResultFormatter:
    """Base class for result formatters."""

    def can_format(self, data: Any) -> bool:
        """Check if this formatter can handle the given data structure."""
        raise NotImplementedError

    def format(self, data: Any) -> str:
        """Format the data for user presentation."""
        raise NotImplementedError


class CalendarEventsFormatter(ResultFormatter):
    """Formatter for calendar event lists."""

    def can_format(self, data: Any) -> bool:
        """Check if data contains calendar events."""
        if not isinstance(data, dict):
            return False
        return "events" in data and isinstance(data.get("events"), list)

    def format(self, data: Dict) -> str:
        """Format calendar events as readable text."""
        events = data.get("events", [])
        event_count = len(events)

        if event_count == 0:
            return "No events found for that time period."

        events_text = []
        for event in events:
            summary = event.get("summary", "Untitled Event")
            start = event.get("start", {}).get("dateTime", "Unknown time")
            events_text.append(f"• {summary} at {start}")

        return f"Found {event_count} event(s):\n" + "\n".join(events_text)


class SearchResultsFormatter(ResultFormatter):
    """Formatter for search results."""

    def can_format(self, data: Any) -> bool:
        """Check if data contains search results."""
        if not isinstance(data, dict):
            return False
        # Look for common search result patterns
        return any(key in data for key in ["results", "items", "hits", "documents"])

    def format(self, data: Dict) -> str:
        """Format search results as readable text."""
        # Find the results array
        results_key = next(
            (key for key in ["results", "items", "hits", "documents"] if key in data),
            None
        )
        if not results_key:
            return json.dumps(data, indent=2)

        results = data[results_key]
        if not results:
            return "No results found."

        formatted = [f"Found {len(results)} result(s):\n"]
        for i, result in enumerate(results[:5], 1):  # Limit to 5 results
            title = result.get("title", result.get("name", "Untitled"))
            url = result.get("url", result.get("link", ""))
            snippet = result.get("snippet", result.get("description", ""))

            formatted.append(f"{i}. {title}")
            if url:
                formatted.append(f"   {url}")
            if snippet:
                formatted.append(f"   {snippet}")

        if len(results) > 5:
            formatted.append(f"\n... and {len(results) - 5} more")

        return "\n".join(formatted)


class GenericJSONFormatter(ResultFormatter):
    """Fallback formatter for generic JSON data."""

    def can_format(self, data: Any) -> bool:
        """Can always format any data."""
        return True

    def format(self, data: Any) -> str:
        """Format as pretty-printed JSON."""
        try:
            return json.dumps(data, indent=2)
        except:
            return str(data)


class FormatterRegistry:
    """Registry of result formatters with priority-based selection."""

    def __init__(self):
        self.formatters = []
        self._register_default_formatters()

    def _register_default_formatters(self):
        """Register built-in formatters in priority order."""
        # More specific formatters first
        self.register(CalendarEventsFormatter())
        self.register(SearchResultsFormatter())
        # Generic fallback last
        self.register(GenericJSONFormatter())

    def register(self, formatter: ResultFormatter):
        """Register a new formatter."""
        self.formatters.append(formatter)

    def format(self, data: Any) -> str:
        """
        Format data using the first applicable formatter.
        Falls back to generic formatter if no specific one matches.
        """
        for formatter in self.formatters:
            if formatter.can_format(data):
                return formatter.format(data)

        # Fallback (should never reach here if GenericJSONFormatter is registered)
        return str(data)


# Global registry instance
_registry = FormatterRegistry()


def get_formatter_registry() -> FormatterRegistry:
    """Get the global formatter registry."""
    return _registry


def format_tool_result(data: Any) -> str:
    """
    Format tool result using registered formatters.
    Convenience function for common use case.
    """
    return _registry.format(data)
