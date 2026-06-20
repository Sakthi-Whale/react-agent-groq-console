"""
app.py
======
Streamlit frontend for the custom ReAct AI Agent.

This file is purely presentational / orchestration glue:
- It builds the page (sidebar + chat area + trace panels).
- It manages Streamlit session state (chat history, stats).
- It calls into `Agent.run()` (agent.py) to get answers.

It does NOT contain any agent reasoning logic - that all lives in agent.py,
unchanged from the original notebook. This file only decides *how to
display* what the agent produces.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from agent import Agent, AgentResult, StepRecord
from config import settings
from tools import get_default_tools
from utils.helpers import (
    conversation_to_json,
    conversation_to_text,
    format_timestamp,
    safe_truncate,
    steps_to_markdown,
)

# --------------------------------------------------------------------------
# Page configuration - must be the first Streamlit call in the script.
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Styling: inject the custom dark-theme CSS from assets/styles.css.
# --------------------------------------------------------------------------
def load_css(css_path: Path) -> None:
    """
    Reads the stylesheet from disk and injects it into the page.

    Uses st.components.v1.html (an isolated iframe technique) combined
    with a parent-document style injection, bypassing st.markdown's HTML
    sanitizer entirely - this avoids environments where unsafe_allow_html
    fails to parse <style> blocks and prints them as visible text instead.
    """
    import streamlit.components.v1 as components

    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        # The trick: render a tiny invisible iframe whose only job is to
        # run JS that reaches OUT of the iframe into the parent document
        # and injects a real <style> tag there. This guarantees the CSS
        # lands in the actual page <head>, not as markdown text.
        components.html(
            f"""
            <script>
                const style = window.parent.document.createElement('style');
                style.innerHTML = `{css}`;
                window.parent.document.head.appendChild(style);
            </script>
            """,
            height=0,
            width=0,
        )
    else:
        st.warning(f"Stylesheet not found at {css_path} - using default styling.")

# --------------------------------------------------------------------------
# Session state initialization.
#
# Streamlit reruns this entire script on every interaction, so anything
# that needs to persist across reruns (chat history, the Agent instance,
# running totals) must live in st.session_state rather than as a plain
# local variable.
# --------------------------------------------------------------------------
def init_session_state() -> None:
    """Sets up every key the app relies on, but only on first run."""

    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "timestamp": str, "steps": list[dict] (assistant only)}
        st.session_state.messages: List[Dict[str, Any]] = []

    if "agent" not in st.session_state:
        # The Agent is expensive to misconfigure repeatedly, so we build it
        # once and reuse it across reruns. It's stateless between calls to
        # `.run()` aside from its own internal `self.state`, which it
        # already clears at the top of every run - so reuse is safe.
        st.session_state.agent = None

    if "stats" not in st.session_state:
        st.session_state.stats: Dict[str, int] = {
            "total_queries": 0,
            "total_tool_calls": 0,
            "total_steps": 0,
        }

    if "config_errors" not in st.session_state:
        st.session_state.config_errors = settings.validate()


init_session_state()


def get_or_create_agent() -> Agent:
    """
    Lazily builds the Agent on first use and caches it in session state.

    Building tools fresh here (rather than once at import time) means the
    sidebar can always show an accurate, current tool list even if config
    changes mid-session (e.g. a user edits .env and restarts).
    """
    if st.session_state.agent is None:
        st.session_state.agent = Agent(tools=get_default_tools())
    return st.session_state.agent


# --------------------------------------------------------------------------
# Sidebar: branding, status, tools, session stats, controls.
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        # ---- Brand header ----
        logo_path = Path(__file__).parent / "assets" / "logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=56)

        st.markdown(
            f"""
            <div class="brand-header">
                <div>
                    <p class="brand-title">{settings.app_title}</p>
                    <p class="brand-sub">{settings.app_tagline}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Agent status ----
        is_configured = len(st.session_state.config_errors) == 0
        status_class = "online" if is_configured else "offline"
        status_text = "Agent Ready" if is_configured else "Not Configured"

        st.markdown(
            f"""
            <div class="sidebar-card">
                <h4>Agent Status</h4>
                <span class="status-pill {status_class}">
                    <span class="status-dot"></span> {status_text}
                </span>
                <div style="margin-top:8px; font-size:0.78rem; color:var(--color-text-muted);">
                    Model: <strong style="color:var(--color-text);">{settings.model_name}</strong><br/>
                    Max steps: <strong style="color:var(--color-text);">{settings.max_steps}</strong><br/>
                    Max uses/tool: <strong style="color:var(--color-text);">{settings.max_uses_per_tool}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not is_configured:
            for problem in st.session_state.config_errors:
                st.error(problem, icon="⚠️")

        # ---- Available tools ----
        tools_html = '<div class="sidebar-card"><h4>Available Tools</h4>'
        for tool in get_default_tools():
            tools_html += f"""
                <div class="tool-chip">
                    <span class="tool-name">🔧 {tool.name}</span>
                    <span class="tool-desc">{tool.description}</span>
                </div>
            """
        tools_html += "</div>"
        st.markdown(tools_html, unsafe_allow_html=True)

        # ---- Session statistics ----
        stats = st.session_state.stats
        st.markdown(
            f"""
            <div class="sidebar-card">
                <h4>Session Statistics</h4>
                <div class="stat-row">
                    <span class="stat-label">Queries answered</span>
                    <span class="stat-value">{stats['total_queries']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Tool calls made</span>
                    <span class="stat-value">{stats['total_tool_calls']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Reasoning steps</span>
                    <span class="stat-value">{stats['total_steps']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Messages in chat</span>
                    <span class="stat-value">{len(st.session_state.messages)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Controls ----
        st.markdown('<div class="sidebar-card"><h4>Controls</h4></div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.stats = {
                "total_queries": 0,
                "total_tool_calls": 0,
                "total_steps": 0,
            }
            st.rerun()

        if st.session_state.messages:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ TXT",
                    data=conversation_to_text(st.session_state.messages),
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "⬇️ JSON",
                    data=conversation_to_json(st.session_state.messages),
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.markdown(
            """
            <div style="margin-top:18px; text-align:center; font-size:0.7rem; color:var(--color-text-muted);">
                Built with a custom ReAct agent · Powered by Groq
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Page header (top of main area).
# --------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="title-block">
                <h1>{settings.app_icon} {settings.app_title}</h1>
                <p>Ask a question - the agent will reason, call tools, and answer.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Chat message rendering.
# --------------------------------------------------------------------------
def render_message(message: Dict[str, Any]) -> None:
    """
    Renders a single chat message as a styled bubble, plus (for assistant
    messages that used tools) an expandable reasoning trace and a copy
    button.
    """
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")

    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
    row_class = "user" if role == "user" else "assistant"
    icon = "🧑" if role == "user" else "🤖"

    st.markdown(
        f"""
        <div class="chat-row {row_class}">
            <div>
                <div class="chat-bubble {bubble_class}">
                    {icon} {content}
                </div>
                <div class="chat-meta">{timestamp}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Assistant messages may carry a `steps` list (the ReAct trace) -
    # render it as an expandable debugging / transparency panel.
    steps = message.get("steps")
    if role == "assistant" and steps:
        with st.expander(f"🔍 View reasoning trace ({len(steps)} step(s))"):
            render_trace(steps)

    # Copy button for assistant text answers - Streamlit has no native
    # clipboard API, so this uses a tiny inline JS snippet via st.markdown.
    if role == "assistant":
        render_copy_button(content, key=f"copy_{timestamp}_{hash(content) % 10_000}")


def render_trace(steps: List[Dict[str, Any]]) -> None:
    """
    Renders the Thought -> Action -> Observation -> Final Answer trace for
    one assistant turn, styled as a sequence of distinct trace-step cards.
    """
    for i, step in enumerate(steps, start=1):
        is_final = step.get("tool") == "Final Answer"
        css_class = "trace-step final" if is_final else "trace-step"
        label = "Final Answer" if is_final else f"Step {i} · {step.get('tool', 'Unknown')}"

        observation_html = ""
        if not is_final:
            observation_html = (
                f'<div style="margin-top:6px;"><strong>Observation:</strong> '
                f"{safe_truncate(step.get('observation', ''), max_chars=1200)}</div>"
            )

        st.markdown(
            f"""
            <div class="{css_class}">
                <div class="trace-label">{label}</div>
                <div style="margin-top:6px;"><strong>Thought:</strong> {step.get('reason', '')}</div>
                <div style="margin-top:4px;"><strong>Action Input:</strong> {step.get('input', '')}</div>
                {observation_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_copy_button(text: str, key: str) -> None:
    """
    Shows the assistant's response in a small text box with Streamlit's
    built-in copy-to-clipboard icon (hover over the top-right corner of
    the code box). This avoids injecting raw HTML/JS, which was rendering
    as visible text instead of an actual button in this environment.
    """
    st.code(text, language=None)


def render_typing_indicator() -> None:
    """Animated 'agent is thinking' indicator shown while a run is in progress."""
    st.markdown(
        """
        <div class="chat-row assistant">
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Core interaction: run the agent against a new user query.
# --------------------------------------------------------------------------
def handle_user_query(query: str) -> None:
    """
    Appends the user's message to history, runs the agent (with a live
    "thinking" placeholder and a step-by-step trace panel that fills in
    as the agent works), then appends the assistant's final answer.
    """
    timestamp = format_timestamp()
    st.session_state.messages.append(
        {"role": "user", "content": query, "timestamp": timestamp}
    )

    # Render the user's bubble immediately so it doesn't wait for the
    # (potentially slow) agent run to complete.
    render_message(st.session_state.messages[-1])

    # Placeholder area where we'll show live progress, then replace it
    # with the final rendered assistant message once done.
    progress_placeholder = st.empty()
    live_steps: List[Dict[str, Any]] = []

    def on_step(step_number: int, step: StepRecord) -> None:
        """
        Callback passed into Agent.run() - invoked after every ReAct step.
        Updates the live trace panel so the user sees Thought/Action/
        Observation appear progressively instead of only at the very end.
        """
        live_steps.append(step.__dict__)
        with progress_placeholder.container():
            render_typing_indicator()
            with st.expander(f"⚙️ Agent is working… ({len(live_steps)} step(s) so far)", expanded=True):
                render_trace(live_steps)

    agent = get_or_create_agent()

    start_time = time.time()
    try:
        result: AgentResult = agent.run(query, on_step=on_step)
    except Exception as exc:
        # Catches anything not already handled inside agent.py (e.g. a
        # misconfigured client raising on construction) so the UI shows a
        # readable error bubble instead of a crashed app.
        progress_placeholder.empty()
        error_text = f"⚠️ The agent encountered an error: {exc}"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_text,
                "timestamp": format_timestamp(),
                "steps": [],
            }
        )
        render_message(st.session_state.messages[-1])
        return

    elapsed = time.time() - start_time
    progress_placeholder.empty()

    # ---- Update running session statistics ----
    tool_call_count = sum(1 for s in result.steps if s.tool != "Final Answer")
    st.session_state.stats["total_queries"] += 1
    st.session_state.stats["total_tool_calls"] += tool_call_count
    st.session_state.stats["total_steps"] += len(result.steps)

    # ---- Determine the final answer text to display ----
    if result.final_answer is not None:
        answer_text = result.final_answer
    elif result.stopped_reason == "parse_error":
        answer_text = (
            "⚠️ The agent's response couldn't be parsed as valid JSON, so it "
            "stopped early. Try rephrasing your question."
        )
    elif result.stopped_reason == "llm_error":
        answer_text = (
            "⚠️ Couldn't reach the language model. Check your GROQ_API_KEY "
            "and network connection, then try again."
        )
    else:
        answer_text = (
            "⚠️ The agent reached its maximum number of steps without "
            "producing a final answer. Here's what it found so far - see "
            "the reasoning trace below for details."
        )

    answer_text += f"\n\n<span style='font-size:0.7rem;color:var(--color-text-muted);'>" \
                    f"Resolved in {elapsed:.1f}s · {len(result.steps)} step(s)</span>"

    assistant_message = {
        "role": "assistant",
        "content": answer_text,
        "timestamp": format_timestamp(),
        "steps": [s.__dict__ for s in result.steps],
    }
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)


# --------------------------------------------------------------------------
# Main page assembly.
# --------------------------------------------------------------------------
def main() -> None:
    render_sidebar()
    render_header()

    # Replay full chat history on every rerun (Streamlit has no persistent
    # DOM between reruns - the whole script re-executes top to bottom).
    for message in st.session_state.messages:
        render_message(message)

    # If config is broken (no API key), block input but keep the UI visible
    # so the user can see exactly what's wrong via the sidebar error banner.
    if st.session_state.config_errors:
        st.info(
            "Chat input is disabled until configuration issues are resolved. "
            "See the sidebar for details.",
            icon="🛑",
        )
        return

    query = st.chat_input("Ask the agent anything…")
    if query:
        handle_user_query(query)


if __name__ == "__main__":
    main()
