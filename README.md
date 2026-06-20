# 🤖 AI Agent Console

A production-ready Streamlit application around a **fully custom, from-scratch ReAct AI agent** — no LangChain agent executor, no LangGraph, no third-party agent framework. The reasoning loop (Reason → Act → Observe → repeat) is hand-written Python that talks directly to the Groq API.

This project is a restructured, UI-wrapped version of an original Jupyter notebook prototype. **The agent's architecture, prompts, tool design, and control flow are unchanged** — only the project layout and the user interface around it are new.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Deployment Guide](#deployment-guide)
- [How the Custom ReAct Agent Works](#how-the-custom-react-agent-works)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)

---

## Overview

The agent answers user questions by reasoning step-by-step, deciding at each step whether to:

1. Call a **tool** (currently `Calculator` or `Search`), observe the result, and continue reasoning, or
2. Produce a **Final Answer** once it has enough information.

This is the classic **ReAct pattern** (Yao et al., 2022) — "Reasoning + Acting" — implemented here as plain Python: a `Tool` base class, two concrete tools, and an `Agent` class that runs a bounded loop, parses strict JSON responses from the LLM, and enforces hard safety caps (max steps, max uses per tool, duplicate-call detection) in code rather than only via prompting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Streamlit UI (app.py)                      │
│                                                                       │
│   Sidebar               Main Chat Area            Trace Panel       │
│   ┌──────────┐         ┌─────────────────┐       ┌───────────────┐ │
│   │ Status   │         │ User message    │       │ Step 1         │ │
│   │ Tools    │         │ Assistant reply │  ───▶ │  Thought       │ │
│   │ Stats    │         │ Typing anim.    │       │  Action        │ │
│   │ Controls │         │                 │       │  Observation   │ │
│   └──────────┘         └─────────────────┘       └───────────────┘ │
└───────────────────────────────┬───────────────────────────────────┘
                                 │  Agent.run(query, on_step=callback)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent (agent.py)                             │
│                                                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  for step in range(max_steps):                               │   │
│   │      1. Build prompt (system + scratchpad)  ── prompts.py    │   │
│   │      2. Call Groq LLM                       ── groq SDK      │   │
│   │      3. Parse JSON {reason, tool, input}                     │   │
│   │      4. If tool == "Final Answer": return                    │   │
│   │      5. Else: check dup/cap guards, then tool.run(input)     │   │
│   │      6. Append Thought/Action/Observation to scratchpad      │   │
│   └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬────────────────────────────────────┘
                                 │  tool.run(input)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Tools (tools.py)                            │
│                                                                        │
│    CalculatorTool                       SearchTool                   │
│    eval() against math namespace        DuckDuckGoSearchRun()        │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design principle:** the Streamlit layer never makes reasoning decisions. It only calls `Agent.run()` and renders whatever comes back. All "intelligence" lives in `agent.py`, `prompts.py`, and `tools.py`.

---

## Features

### Core Agent (preserved from the original notebook)
- Custom ReAct loop — Reason → Act → Observe, no agent framework
- Strict JSON step protocol between the LLM and the control loop
- Hard-coded safety limits: max steps per run, max uses per tool, duplicate-call detection
- Pluggable `Tool` base class — add new tools by subclassing and implementing `.run()`

### Frontend (new)
- Dark, modern "AI SaaS" theme — deep indigo + cyan accents on a slate background
- Custom chat bubbles for user/assistant turns with timestamps
- Live typing/"agent is working" indicator while the agent reasons
- **Expandable reasoning trace panel** showing Thought → Action → Observation → Final Answer for every turn
- Copy-response button on every assistant message
- Download full conversation as `.txt` or `.json`
- Sidebar with agent status, tool catalog, live session statistics, and a clear-conversation control
- Friendly configuration-error banner instead of a stack trace if `GROQ_API_KEY` is missing

### Engineering
- Fully modular: `config.py`, `prompts.py`, `tools.py`, `agent.py`, `utils/helpers.py`, `app.py`
- Type hints throughout
- Centralized environment variable handling via `config.py`
- Defensive error handling around the LLM call and JSON parsing (the original notebook would crash; this version degrades gracefully)
- Deployment-ready (`requirements.txt`, `.env.example`, no hardcoded secrets)

---

## Project Structure

```
project/
│
├── app.py                # Streamlit frontend — layout, session state, rendering
├── agent.py               # Custom Agent class — the ReAct loop (core logic, preserved)
├── tools.py                # Tool base class + CalculatorTool + SearchTool
├── prompts.py             # System prompt + per-step user prompt templates
├── config.py               # Centralized settings, reads from environment variables
├── requirements.txt    # Python dependencies
├── .env.example          # Template for required environment variables
├── README.md             # This file
│
├── assets/
│   ├── logo.png            # App logo shown in the sidebar
│   └── styles.css          # Custom dark theme CSS
│
└── utils/
    ├── __init__.py
    └── helpers.py          # Formatting, export (txt/json), timestamp utilities
```

---

## Installation

### Prerequisites
- Python 3.9+
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### Steps

```bash
# 1. Clone or copy the project, then move into it
cd project

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and paste your real GROQ_API_KEY
```

---

## Configuration

All configuration is environment-variable driven (see `config.py`). Set these in your `.env` file or directly in your deployment platform's secret manager:

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model identifier |
| `GROQ_TEMPERATURE` | No | `0` | LLM sampling temperature |
| `MAX_STEPS` | No | `10` | Max ReAct loop iterations per query |
| `MAX_USES_PER_TOOL` | No | `2` | Hard cap on calls to any one tool per run |
| `SEARCH_MAX_RESULTS` | No | `5` | Reserved for search tool tuning |

If `GROQ_API_KEY` is missing, the app still launches but disables chat input and shows a clear warning in the sidebar instead of crashing.

---

## Running Locally

```bash
streamlit run app.py
```

By default this opens the app at `http://localhost:8501`.

---

## Deployment Guide

### Option A — Streamlit Community Cloud
1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing at `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. Streamlit Cloud injects secrets as environment variables automatically, which `config.py` picks up via `os.getenv`.

### Option B — Docker
Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ai-agent-console .
docker run -p 8501:8501 --env-file .env ai-agent-console
```

### Option C — Render / Railway / Fly.io / Heroku-style PaaS
- Set the start command to: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Add `GROQ_API_KEY` (and any other overrides) as platform environment variables/secrets — never commit `.env`.

### Production Notes
- `config.py` validates configuration on startup; check sidebar warnings first if something looks broken post-deploy.
- The `Search` tool makes outbound calls to DuckDuckGo — ensure your deployment environment allows outbound HTTPS.
- Consider adding rate limiting in front of the app if exposing it publicly, since each chat turn consumes Groq API quota.

---

## How the Custom ReAct Agent Works

This is **not** a LangChain `AgentExecutor` or a LangGraph state machine. It's a plain Python loop:

1. **System prompt construction** (`prompts.build_system_prompt`): lists every registered tool by name and description, plus a "Final Answer" pseudo-tool, and instructs the LLM to always respond with one of two strict JSON shapes — either `{"reason", "tool", "input"}` for a tool call, or the same shape with `"tool": "Final Answer"` to finish.

2. **The loop** (`Agent.run`): for up to `max_steps` iterations:
   - Builds a user-turn message containing the original query, the running scratchpad of prior Thought/Action/Observation text, and the current step number.
   - Sends `[system_prompt, user_turn]` to Groq with `temperature=0` for deterministic, parseable output.
   - Parses the response as JSON. If parsing fails, the loop stops cleanly (`stopped_reason = "parse_error"`) rather than guessing.
   - If `tool == "Final Answer"`, the loop returns immediately with the answer.
   - Otherwise, before calling the tool, two **hard guards** run in code (not just prompted):
     - **Duplicate-call detection** — if this exact `(tool, input)` pair was already used this run, the tool is *not* re-invoked; instead the agent is told it already has that observation.
     - **Per-tool usage cap** — once a tool has been used `max_uses_per_tool` times, further calls to it are blocked with an explanatory observation, nudging the agent toward a different tool or a Final Answer.
   - The tool's result becomes the "Observation" appended to the scratchpad for the next iteration.

3. **Tools** (`tools.py`):
   - `CalculatorTool` evaluates Python expressions against a namespace built from the `math` module (`sqrt`, `pow`, `pi`, …), so both `sqrt(81)` and `math.sqrt(81)` work. Any exception returns the string `"Calculation Error"` rather than raising.
   - `SearchTool` wraps LangChain Community's `DuckDuckGoSearchRun` purely as an HTTP search utility — using this one helper class does not make the agent itself LangChain-based; it's the same role `requests` would play if DuckDuckGo had a simpler public API.

4. **State** (`Agent.state` / `AgentResult`): every step (Thought, Action, Action Input, Observation) is recorded both in a flat dict (`Agent.state`, matching the original notebook's shape) and in a typed `AgentResult` object that the UI consumes directly for rendering the trace panel.

---

## Screenshots

> Replace these placeholders with real screenshots once you've run the app locally.

| Chat View | Reasoning Trace |
|---|---|
| `docs/screenshots/chat-view.png` | `docs/screenshots/trace-panel.png` |

| Sidebar | Dark Theme Overview |
|---|---|
| `docs/screenshots/sidebar.png` | `docs/screenshots/full-app.png` |

---

## Troubleshooting

**"GROQ_API_KEY is not set" banner won't go away**
Make sure `.env` is in the project root (same folder as `app.py`) and the key has no surrounding quotes or extra whitespace.

**Search tool returns errors**
DuckDuckGo's unofficial search endpoint occasionally rate-limits. The error is surfaced as an Observation (`"Search Error: ..."`) so the agent can reason around it rather than crashing the app.

**Agent stops with "reached max_steps without a Final Answer"**
Increase `MAX_STEPS` in `.env`, or simplify the query — this is the same safety behavior as the original notebook, just surfaced more clearly in the UI.

**Styling looks off / default Streamlit theme shows through**
Confirm `assets/styles.css` exists and is readable; `app.py` will print a warning if it can't find the stylesheet.
"# react-agent-groq-console" 
