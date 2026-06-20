"""
config.py
=========
Centralized configuration for the agent application.

Why this file exists:
- Keeps every "magic value" (model name, API keys, limits) in ONE place
  instead of scattered across agent.py / tools.py / app.py.
- Reads secrets from environment variables instead of hardcoding them,
  so the same code works in local dev, Docker, and cloud deployments
  without any source changes.
- Other modules import the shared `settings` object from here, so there is
  a single source of truth for configuration.
"""

import os
from dataclasses import dataclass, field
from typing import List

# load_dotenv() reads a local .env file (if one exists) and copies its
# KEY=VALUE pairs into os.environ. In production (Streamlit Cloud, Docker,
# etc.) the platform usually injects env vars directly, so this call is a
# harmless no-op there - it only matters for local development.
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings container.

    frozen=True makes every attribute read-only after the object is built.
    This is intentional: configuration should be set once at startup and
    never silently mutated mid-run, which would make bugs very hard to trace.
    """

    # ---------------- Groq / LLM configuration ----------------

    # Pulled from the environment so the real key is never committed to
    # source control. Defaults to "" (empty) if missing; we surface that
    # as a friendly validation error rather than a crash (see validate()).
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # The exact model string the Groq SDK expects. Stored here (not inside
    # agent.py) so it can be swapped via an env var with zero code changes.
    # Defaults to the same model used in the original notebook.
    model_name: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    )

    # Sampling temperature sent to the LLM. 0 = fully deterministic output,
    # which matters here because the agent expects strict JSON back at every
    # step - creative/random variation would break the parser.
    temperature: float = field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0"))
    )

    # ---------------- Agent loop configuration ----------------

    # Maximum number of Reason -> Act -> Observe iterations before the agent
    # is forced to stop even without a Final Answer. Prevents infinite loops
    # (and runaway API costs) if the LLM never converges.
    max_steps: int = field(default_factory=lambda: int(os.getenv("MAX_STEPS", "10")))

    # Hard cap on how many times any single tool may be called within one
    # run. This mirrors the original notebook's `max_uses_per_tool = 2` and
    # is enforced in code (agent.py), not just requested via the prompt.
    max_uses_per_tool: int = field(
        default_factory=lambda: int(os.getenv("MAX_USES_PER_TOOL", "2"))
    )

    # ---------------- App / UI metadata ----------------

    app_title: str = "AI Agent Console"
    app_icon: str = "🤖"
    app_tagline: str = "ReAct-style reasoning agent powered by Groq"

    # ---------------- Search tool configuration ----------------

    # Reserved for future tuning of the search tool (e.g. result count).
    # Exposed here so behavior can be tweaked without editing tools.py.
    search_max_results: int = field(
        default_factory=lambda: int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    )

    def validate(self) -> List[str]:
        """
        Checks the current configuration for problems and returns them as a
        list of human-readable strings. An empty list means "all good".

        Why return a list instead of raising an exception:
        Streamlit needs to render a friendly warning banner in the UI rather
        than crash with a raw traceback. By returning problems instead of
        raising, app.py can decide exactly how to display them to the user.
        """
        problems: List[str] = []

        if not self.groq_api_key:
            problems.append(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or set the environment variable before running."
            )

        if self.max_steps < 1:
            problems.append("MAX_STEPS must be at least 1.")

        if self.max_uses_per_tool < 1:
            problems.append("MAX_USES_PER_TOOL must be at least 1.")

        return problems


# A single shared instance, imported elsewhere as `from config import settings`.
# This "settings singleton" pattern avoids re-reading environment variables
# or re-instantiating the dataclass in multiple places.
settings = Settings()
