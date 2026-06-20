"""
agent.py
========
The custom ReAct-style agent, extracted from the notebook.

IMPORTANT: This file preserves the original agent architecture exactly -
the Reason -> Act -> Observe loop, the JSON step protocol, the per-tool
usage cap, the duplicate-call detection, and the scratchpad accumulation
are all unchanged in behavior from the notebook. Changes made here are
purely structural / cosmetic:
  - type hints added
  - docstrings added
  - print() statements replaced with a pluggable callback so the Streamlit
    UI can render steps live instead of only seeing console output
  - defensive error handling added around the Groq API call and JSON parsing
  - state is returned in a structured form (StepRecord / AgentResult) rather
    than only being stored on a plain dict, which makes it easy to render
    in Streamlit without re-parsing strings

No reasoning logic, prompt content, or control flow was changed.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from groq import Groq

from config import settings
from prompts import build_system_prompt, build_user_turn_prompt
from tools import Tool


@dataclass
class StepRecord:
    """
    A single Reason -> Act -> Observe step, recorded for both the
    in-memory `Agent.state` history and for rendering in the UI.

    Mirrors the dict shape the notebook appended to `self.state["steps"]`:
    {"reason": ..., "tool": ..., "input": ..., "observation": ...}
    but as a typed dataclass instead of a loose dict, which avoids typos
    like `step["reasom"]` when this gets consumed elsewhere.
    """

    reason: str
    tool: str
    input: str
    observation: Any


@dataclass
class AgentResult:
    """
    The full outcome of one `Agent.run()` call.

    Bundles the original query, every step taken, and the final answer (or
    None if the agent ran out of steps) into a single typed object that
    app.py can render without reaching into private attributes.
    """

    query: str
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    stopped_reason: Optional[str] = None  # "final_answer" | "max_steps" | "parse_error"


# A callback the Agent can invoke after every step so callers (like the
# Streamlit app) can render progress live instead of waiting for the whole
# run to finish. Signature: callback(step_number: int, step: StepRecord) -> None
StepCallback = Callable[[int, StepRecord], None]


class Agent:
    """
    A ReAct-style agent: loops Reason -> Act -> Observe until it produces a
    Final Answer or runs out of steps. Each instance owns its own client,
    tools, and state - no globals - exactly as designed in the notebook.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_uses_per_tool: Optional[int] = None,
    ) -> None:
        """
        Args:
            tools: list of Tool instances available to the agent this run.
            model: Groq model name; defaults to the configured setting so
                   callers don't have to repeat it everywhere.
            max_steps: cap on ReAct loop iterations; defaults to settings.
            max_uses_per_tool: hard cap on calls per tool; defaults to settings.
        """
        # The Groq client reads its key from config.settings rather than
        # calling os.getenv directly here - keeps all env-var access
        # centralized in config.py.
        self.client = Groq(api_key=settings.groq_api_key)

        self.model = model or settings.model_name
        self.max_steps = max_steps or settings.max_steps

        # Dict keyed by tool name, exactly like the notebook
        # (`{tool.name: tool for tool in tools}`), so lookups by the LLM's
        # chosen tool name are O(1).
        self.tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

        # Hard cap per tool per run, enforced in code (not just prompted) -
        # same safeguard as the notebook.
        self.max_uses_per_tool = max_uses_per_tool or settings.max_uses_per_tool

        # Mirrors the notebook's `self.state` dict, kept for backward
        # compatibility with anything that expects `agent.state["steps"]`.
        self.state: Dict[str, Any] = {}

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        Delegates to prompts.build_system_prompt, which contains the exact
        wording from the notebook's `_build_system_prompt` method. Kept as
        a thin wrapper method (instead of inlining the call at every use
        site) so the public interface of `Agent` matches the notebook 1:1.
        """
        return build_system_prompt(self.tools)

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Sends `messages` to the Groq chat completion endpoint and returns
        the raw text content of the model's reply.

        temperature=0 (via settings) keeps output deterministic, which
        matters because the agent expects strict JSON back at every step.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.temperature,
        )
        return response.choices[0].message.content

    def run(
        self,
        query: str,
        on_step: Optional[StepCallback] = None,
    ) -> AgentResult:
        """
        Executes the ReAct loop for `query` until a Final Answer is produced
        or `max_steps` is reached.

        Args:
            query: the user's question.
            on_step: optional callback invoked after each step is recorded,
                     so a UI can render progress live. This is purely
                     additive - the underlying loop logic is identical to
                     the notebook's `Agent.run`.

        Returns:
            An AgentResult containing every step taken and the final answer
            (or None if the agent stopped without one).
        """
        # Reset state at the start of every run, exactly like the notebook's
        # `self.state.clear()` followed by re-population.
        self.state.clear()
        self.state["query"] = query
        self.state["steps"] = []
        self.state["result"] = None

        result = AgentResult(query=query)

        # The running Thought/Action/Observation transcript fed back into
        # the prompt at every step - identical concept to the notebook's
        # `scratchpad` string.
        scratchpad = ""

        # Tracks how many times each tool name has been used this run.
        tool_usage_count: Dict[str, int] = {}

        # Tracks (tool_name, normalized_input) pairs already attempted, so
        # repeated identical calls can be short-circuited with a warning
        # observation instead of re-hitting the tool (and, for Search,
        # the network) again.
        seen_calls = set()

        for step_number in range(self.max_steps):

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": build_user_turn_prompt(
                        query=query,
                        scratchpad=scratchpad,
                        step_number=step_number,
                        max_steps=self.max_steps,
                    ),
                },
            ]

            try:
                llm_output = self._call_llm(messages)
            except Exception as exc:
                # Network/API errors weren't explicitly handled in the
                # notebook; this addition prevents a crash from taking down
                # the whole Streamlit session, surfacing a clean stop reason
                # instead.
                result.stopped_reason = "llm_error"
                result.final_answer = None
                self.state["result"] = None
                self.state["error"] = f"LLM call failed: {exc}"
                return result

            try:
                step_data = json.loads(llm_output)
            except Exception:
                # Matches the notebook's behavior: if the model doesn't
                # return valid JSON, stop the loop rather than guessing.
                print("Could not parse LLM output as JSON:", llm_output)
                result.stopped_reason = "parse_error"
                self.state["raw_unparsed_output"] = llm_output
                break

            reason = step_data.get("reason", "")
            tool_name = step_data.get("tool", "")
            tool_input = step_data.get("input", "")

            print(f"\n--- Step {step_number + 1} ---")
            print("Reason:", reason)
            print("Tool:", tool_name)
            print("Input:", tool_input)

            # ---- Final Answer branch ----
            if tool_name == "Final Answer":
                step_record = StepRecord(
                    reason=reason, tool=tool_name, input=tool_input, observation="N/A"
                )
                self.state["steps"].append(step_record.__dict__)
                self.state["result"] = tool_input

                result.steps.append(step_record)
                result.final_answer = tool_input
                result.stopped_reason = "final_answer"

                if on_step:
                    on_step(step_number, step_record)

                print("\nFinal Answer:", tool_input)
                return result

            # Normalized signature used to detect exact repeat calls -
            # same logic as the notebook (`str(tool_input).strip().lower()`).
            call_signature = (tool_name, str(tool_input).strip().lower())

            if call_signature in seen_calls:
                observation = (
                    f"You already made this exact '{tool_name}' call with this input earlier in this task - "
                    f"repeating it will return the same result. Try a meaningfully different input, switch "
                    f"to a different tool, or give your Final Answer now using what you already know."
                )

            elif tool_usage_count.get(tool_name, 0) >= self.max_uses_per_tool:
                observation = (
                    f"You have already used '{tool_name}' {tool_usage_count[tool_name]} times in this task, "
                    f"which is the maximum allowed. You may NOT use it again. Use a different tool, or give "
                    f"your Final Answer now with your best estimate from what you've already gathered."
                )

            else:
                tool = self.tools.get(tool_name)

                if tool is None:
                    observation = "Unknown tool requested."
                else:
                    observation = tool.run(tool_input)
                    tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
                    seen_calls.add(call_signature)

            print("Observation:", observation)

            step_record = StepRecord(
                reason=reason, tool=tool_name, input=tool_input, observation=observation
            )
            self.state["steps"].append(step_record.__dict__)
            result.steps.append(step_record)

            if on_step:
                on_step(step_number, step_record)

            # Append this step to the scratchpad exactly as the notebook
            # does, so the next LLM call sees the full Thought/Action/
            # Observation history in the same textual format.
            scratchpad += (
                f"\nThought: {reason}\nAction: {tool_name}\n"
                f"Action Input: {tool_input}\nObservation: {observation}\n"
            )

        print("\nAgent stopped: reached max_steps without a Final Answer.")
        if result.stopped_reason is None:
            result.stopped_reason = "max_steps"
        self.state.setdefault("result", None)
        return result
