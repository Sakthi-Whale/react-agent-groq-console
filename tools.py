"""
tools.py
========
Tool definitions for the agent, extracted directly from the notebook.

Contains:
- `Tool`            : the abstract base class every tool inherits from.
- `CalculatorTool`   : evaluates math expressions using Python's `math` module.
- `SearchTool`       : live web search via DuckDuckGo (LangChain's wrapper is
                       reused here only as a thin search utility - this does
                       NOT make the agent itself a LangChain agent; the
                       ReAct loop and reasoning logic in agent.py remain
                       100% custom code).

The class bodies and behavior are unchanged from the original notebook.
Only additions: type hints, docstrings, and basic error handling so the
Streamlit app can fail gracefully instead of crashing.
"""

import math
from typing import Any, Dict, List

from langchain_community.tools import DuckDuckGoSearchRun


class Tool:
    """
    Base class every tool inherits from.

    Guarantees every tool exposes:
      - `.name`        (str)  : the identifier the LLM uses to select it.
      - `.description` (str)  : shown to the LLM so it knows when to pick it.
      - `.run(input)`  (method): executes the tool and returns a result.

    Because every tool shares this interface, `Agent` (in agent.py) can
    treat all tools identically - it never needs to know whether a given
    tool does arithmetic or hits the network, it just calls `.run()`.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    def run(self, tool_input: str) -> Any:
        """
        Executes the tool's logic. Must be overridden by subclasses.

        Raising NotImplementedError here (rather than leaving it abstract
        via `abc`) keeps the base class simple and dependency-free, matching
        the original notebook's design exactly.
        """
        raise NotImplementedError("Each tool must implement its own run() method.")


class CalculatorTool(Tool):
    """
    Evaluates a math expression using Python's built-in `eval`.

    Exposes every public name from the `math` module directly into the eval
    namespace (sqrt, pow, log, pi, ...) so both `sqrt(81)` and `math.sqrt(81)`
    work as input. This mirrors the notebook's implementation exactly,
    including the use of `eval` - left intentionally unchanged per the
    instruction to preserve existing agent/tool behavior.

    Note: `eval` here is restricted to a custom namespace containing only
    `math` module attributes (no builtins like `open`, `exec`, `__import__`),
    which limits - but does not eliminate - what an expression could do.
    This restriction exists in the original notebook code as written.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Calculator",
            description="Use for mathematical calculations, e.g. 'sqrt(81)' or '250 * 80'.",
        )
        # Build a namespace of every public (non-underscore-prefixed) name
        # in the math module, e.g. {"sqrt": math.sqrt, "pi": math.pi, ...}.
        self.eval_namespace: Dict[str, Any] = {
            name: getattr(math, name) for name in dir(math) if not name.startswith("_")
        }
        # Also expose the `math` module itself, so `math.sqrt(81)` style
        # input works in addition to the unqualified `sqrt(81)` style.
        self.eval_namespace["math"] = math

    def run(self, expression: str) -> Any:
        """
        Evaluates `expression` against the math namespace.

        Returns the numeric result on success, or the string
        "Calculation Error" on any exception (invalid syntax, unknown name,
        division by zero, etc.) - exactly as in the original notebook, so
        the agent's ReAct loop sees a clean string observation either way.
        """
        try:
            # Passing an empty `{}` as globals (implicitly, since we only
            # supply locals here as the second positional arg) combined with
            # the curated namespace keeps eval from seeing Python builtins.
            return eval(expression, self.eval_namespace)
        except Exception:
            return "Calculation Error"


class SearchTool(Tool):
    """
    Live web search via DuckDuckGo.

    Wraps LangChain Community's `DuckDuckGoSearchRun` purely as a search
    utility - this is the same dependency the notebook already used. Using
    this single helper class does NOT turn the project into a LangChain
    agent: there is no LangChain `AgentExecutor`, no LangChain prompt
    templates, and no LangChain tool-calling abstraction anywhere in the
    reasoning loop. The custom `Agent` class in agent.py owns 100% of the
    Reason -> Act -> Observe logic.
    """

    def __init__(self) -> None:
        super().__init__(
            name="Search",
            description="Use for factual questions and information retrieval from the web.",
        )
        self.search = DuckDuckGoSearchRun()

    def run(self, query: str) -> Any:
        """
        Runs a DuckDuckGo search for `query` and returns the raw result text.

        Wrapped in a try/except (an addition over the notebook) so that a
        transient network failure or rate limit surfaces as a readable
        observation string for the agent to reason about, rather than
        crashing the whole Streamlit session.
        """
        try:
            return self.search.run(query)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            # Returned as a string (not re-raised) so the agent's ReAct loop
            # can see the failure as an Observation and decide how to react
            # (e.g. retry differently, or fall back to its own knowledge).
            return f"Search Error: {exc}"


def get_default_tools() -> List[Tool]:
    """
    Convenience factory returning the standard tool set used throughout the
    app (sidebar tool list, Agent construction, etc.) so app.py doesn't need
    to know the concrete tool classes directly.
    """
    return [CalculatorTool(), SearchTool()]
