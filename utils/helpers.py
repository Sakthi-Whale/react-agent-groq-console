"""
utils/helpers.py
=================
Small, pure utility functions shared by app.py.

Kept separate from app.py so that:
- UI rendering code (app.py) stays focused on layout/widgets.
- These functions can be unit-tested in isolation without spinning up
  Streamlit.
- Formatting logic isn't duplicated if multiple parts of the UI need the
  same conversation-export or timestamp behavior.
"""

import json
from datetime import datetime
from typing import Any, Dict, List


def format_timestamp(dt: datetime = None) -> str:
    """
    Returns a short, human-readable timestamp string, e.g. "14:32:07".

    Used to stamp each chat message so the conversation log reads like a
    real chat client rather than an undated wall of text.
    """
    dt = dt or datetime.now()
    return dt.strftime("%H:%M:%S")


def steps_to_markdown(steps: List[Dict[str, Any]]) -> str:
    """
    Renders a list of agent step dicts (reason/tool/input/observation) as a
    readable Markdown trace, in the classic
        Thought: ...
        Action: ...
        Action Input: ...
        Observation: ...
    ReAct format. Used both for the expandable "trace" panel in the UI and
    for the downloadable conversation export.

    Args:
        steps: list of dicts shaped like {"reason", "tool", "input",
               "observation"} - the same shape `Agent.state["steps"]`
               stores.

    Returns:
        A single Markdown string with one block per step.
    """
    if not steps:
        return "_No intermediate steps were recorded for this answer._"

    blocks = []
    for i, step in enumerate(steps, start=1):
        is_final = step.get("tool") == "Final Answer"
        header = f"**Step {i} — {'Final Answer' if is_final else step.get('tool', 'Unknown Tool')}**"
        body = (
            f"> **Thought:** {step.get('reason', '')}\n"
            f">\n"
            f"> **Action:** {step.get('tool', '')}\n"
            f">\n"
            f"> **Action Input:** {step.get('input', '')}\n"
        )
        if not is_final:
            body += f">\n> **Observation:** {step.get('observation', '')}\n"
        blocks.append(f"{header}\n\n{body}")

    return "\n\n".join(blocks)


def conversation_to_text(messages: List[Dict[str, Any]]) -> str:
    """
    Serializes the full chat history into a plain-text transcript suitable
    for the "Download conversation" button.

    Each message dict is expected to have at least:
        {"role": "user" | "assistant", "content": str, "timestamp": str}
    Assistant messages may additionally include "steps" (the ReAct trace),
    which - if present - is appended beneath the final answer so the
    downloaded file is a complete record, not just the headline answers.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("AI AGENT CONSOLE — CONVERSATION EXPORT")
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    for msg in messages:
        role_label = "USER" if msg.get("role") == "user" else "ASSISTANT"
        timestamp = msg.get("timestamp", "")
        lines.append(f"[{timestamp}] {role_label}:")
        lines.append(msg.get("content", ""))

        steps = msg.get("steps")
        if steps:
            lines.append("")
            lines.append("-- Reasoning trace --")
            for i, step in enumerate(steps, start=1):
                lines.append(f"  Step {i}")
                lines.append(f"    Thought:     {step.get('reason', '')}")
                lines.append(f"    Action:      {step.get('tool', '')}")
                lines.append(f"    Action Input:{step.get('input', '')}")
                if step.get("tool") != "Final Answer":
                    lines.append(f"    Observation: {step.get('observation', '')}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    return "\n".join(lines)


def conversation_to_json(messages: List[Dict[str, Any]]) -> str:
    """
    Serializes the chat history to a pretty-printed JSON string - an
    alternative export format for users who want machine-readable output
    (e.g. to feed into another tool or archive programmatically).
    """
    payload = {
        "exported_at": datetime.now().isoformat(),
        "messages": messages,
    }
    # default=str guards against any non-JSON-serializable values (rare,
    # but tool observations could theoretically be non-primitive types)
    # ending up in the export and crashing the download.
    return json.dumps(payload, indent=2, default=str)


def safe_truncate(text: str, max_chars: int = 4000) -> str:
    """
    Truncates very long tool observations (e.g. large search results) for
    display purposes, appending a clear marker so users know content was
    cut off. Keeps the UI responsive and readable without losing the full
    data (the untruncated version is still what gets sent back to the LLM
    by agent.py - this helper is purely for rendering).
    """
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n… [truncated, {len(text) - max_chars} more characters]"
