"""
Web tools — let the agent search the web and fetch URLs.

These tools expand the agent's knowledge beyond what's in the training data or
local files. Web search is the most powerful tool for a general-purpose agent.
"""

import os
import json
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.tools import ToolRegistry


def web_search(query: str) -> str:
    """
    Search the web using SerpAPI (Google).

    Requires SERPAPI_API_KEY in environment or .env file.

    Args:
        query: The search query.
    """
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return (
            "Error: SERPAPI_API_KEY not set. To enable web search:\n"
            "1. Get a free key at https://serpapi.com\n"
            "2. Add SERPAPI_API_KEY=your_key to .env"
        )

    try:
        params = {
            "q": query,
            "engine": "google",
            "api_key": api_key,
        }
        # Build URL with query params
        from urllib.parse import urlencode
        url = "https://serpapi.com/search?" + urlencode(params)

        req = Request(url, headers={"User-Agent": "AgentApp/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        # Extract organic results
        results = data.get("organic_results", [])
        if not results:
            # Check for answer box
            answer = data.get("answer_box", {})
            if answer:
                return f"Answer: {answer.get('answer', answer.get('snippet', str(answer)))}"
            return f"No results found for: {query}"

        lines = [f"Search results for '{query}':"]
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "No title")
            snippet = r.get("snippet", "No description")
            link = r.get("link", "")
            lines.append(f"\n{i}. {title}")
            lines.append(f"   {snippet}")
            lines.append(f"   {link}")

        return "\n".join(lines)

    except URLError as e:
        return f"Error: Web search failed: {e}"
    except Exception as e:
        return f"Error: {e}"


def fetch_url(url: str) -> str:
    """
    Fetch and read the content of a web page.

    Args:
        url: The URL to fetch.
    """
    try:
        req = Request(
            url,
            headers={"User-Agent": "AgentApp/1.0"},
        )
        with urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            # Basic HTML-to-text: strip tags for readability
            text = _strip_html(content)
            if len(text) > 10_000:
                text = text[:10_000] + f"\n... (truncated, {len(text)} total chars)"
            return text

    except URLError as e:
        return f"Error: Could not fetch {url}: {e}"
    except Exception as e:
        return f"Error: {e}"


def _strip_html(html: str) -> str:
    """Crude HTML-to-text conversion. Good enough for agent consumption."""
    import re
    # Remove scripts and styles
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    text = re.sub(r'</?(div|p|h[1-6]|li|tr|br|article|section|header|footer)[^>]*>', '\n', text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def create_web_tools() -> ToolRegistry:
    """Create a registry with all web tools."""
    registry = ToolRegistry()

    registry.register(
        name="web_search",
        description=(
            "Search the web using Google. Returns top results with titles, snippets, and links. "
            "Use this when you need current information or facts you're unsure about."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "The search query.",
            }
        },
    )(web_search)

    registry.register(
        name="fetch_url",
        description="Fetch and read the text content of a web page at the given URL.",
        parameters={
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            }
        },
    )(fetch_url)

    return registry
