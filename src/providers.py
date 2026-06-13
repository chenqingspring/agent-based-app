"""
Provider configuration — maps provider IDs to API endpoints and auth.

Inspired by pi's api-registry.ts: each provider is a self-contained config
that the LLM client uses to connect. Adding a new Anthropic-compatible
provider is just adding an entry to PROVIDERS.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    id: str
    name: str                       # Human-readable name
    api_key_env: str                # Environment variable for the API key
    base_url: str | None = None     # Custom endpoint (None = SDK default)
    default_model: str = ""

    def resolve_api_key(self) -> str:
        """Get API key from environment or raise a helpful error."""
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"{self.api_key_env} not set. Add it to your .env file:\n"
                f"  {self.api_key_env}=your-key-here"
            )
        return api_key


# ---------------------------------------------------------------------------
# Registered providers
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url=None,
        default_model="claude-sonnet-4-6",
    ),
    "volcengine": ProviderConfig(
        id="volcengine",
        name="Volcengine ARK",
        api_key_env="VOLCENGINE_API_KEY",
        base_url="https://ark.cn-beijing.volces.com/api/coding",
        default_model="deepseek-v4-pro[1m]",
    ),
}


def get_provider(provider_id: str) -> ProviderConfig:
    """Get a provider config by ID."""
    if provider_id not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider_id}'. Available: {available}")
    return PROVIDERS[provider_id]


def list_providers() -> list[ProviderConfig]:
    """Get all registered providers."""
    return list(PROVIDERS.values())
