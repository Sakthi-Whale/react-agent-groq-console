"""
prompts.py
==========
All prompt text used by the agent, extracted verbatim (in meaning and
wording) from the original notebook's `_build_system_prompt` method.

Why this is a separate file:
- Prompt engineering changes shouldn't require touching agent.py's logic.
- Keeping prompt text in one place makes it easy to review, version, and
  tweak wording without scrolling through control-flow code.
- This module exposes a single function, `build_system_prompt`, which
  agent.py calls - the *content* of the prompt is unchanged from the
  notebook, only its location has moved.
"""

from typing import Dict

from tools import Tool  # noqa: F401  (imported for the type hint below)


# The static "skeleton" of the system prompt. It is a format string with one
# placeholder, {tools_block}, which gets filled in per-agent based on
# whichever tools were actually registered. Keeping this as a module-level
# constant (rather than rebuilding the literal string inside a function)
# makes the prompt easy to find and diff in version control.
_SYSTEM_PROMPT_TEMPLATE = """
You are an AI Agent that thinks step-by-step using the ReAct (Reason + Act) pattern.

Available Tools:

{tools_block}{final_answer_index}. Final Answer
- Use this ONLY when you already have enough information to fully answer the user's question.

You will be shown the conversation so far, including any earlier Thoughts, Actions, and Observations.
At every step, decide what to do next and return ONLY a JSON object (no markdown fences, no extra text)
in ONE of these two formats:

If you still need to use a tool:
{{
    "reason": "why you are taking this step",
    "tool": "the exact tool name",
    "input": "the input to give the tool"
}}

If you already know the final answer:
{{
    "reason": "why you now have enough information",
    "tool": "Final Answer",
    "input": "the final answer to give the user"
}}

Rules:
- Use only ONE tool per step.
- Always check the Observations already shown to you before acting - don't repeat a call you already made.
- Real-world tool results are messy by nature: they mix old and current information, repeat the same fact
  in different words, and often report slightly different numbers from different sources - for example, a
  city's population vs. its district's population vs. its metro area's population, or a figure reported in
  different years or currencies. This is completely normal - do NOT keep using the same tool just to find
  one "perfect" or "exact" answer.
- If a name or fact appears more than once (even worded differently), treat it as confirmed. Each tool can
  only be used twice in this entire task - after that, you must switch tools or give a Final Answer.
- An approximate but timely answer is better than no answer. Give the Final Answer as soon as you reasonably
  can.
- You have a limited number of steps, and you will be told which step you're on. As you approach your last
  step, you MUST wrap up with a Final Answer using whatever information you've already gathered, even if it
  isn't perfect.
"""


def build_system_prompt(tools: Dict[str, "Tool"]) -> str:
    """
    Builds the full system prompt string for a given set of tools.

    This is a direct extraction of `Agent._build_system_prompt` from the
    notebook - same numbering scheme (tools listed 1..N, "Final Answer"
    listed as N+1), same wording, same rules. The only change is that it now
    lives as a standalone function so agent.py can import it instead of
    defining it inline.

    Args:
        tools: dict mapping tool name -> Tool instance (same shape as
               `Agent.tools` in the original notebook).

    Returns:
        The complete system prompt string to send to the LLM.
    """
    tools_block = ""
    for i, tool in enumerate(tools.values(), start=1):
        # Each tool contributes a numbered "name + description" block,
        # exactly matching the notebook's formatting so the LLM sees an
        # identical prompt structure to what it was originally tested with.
        tools_block += f"{i}. {tool.name}\n- {tool.description}\n\n"

    # "Final Answer" is always listed last, numbered one past the last tool -
    # this preserves the exact same indexing the notebook used.
    final_answer_index = len(tools) + 1

    return _SYSTEM_PROMPT_TEMPLATE.format(
        tools_block=tools_block,
        final_answer_index=final_answer_index,
    )


def build_user_turn_prompt(query: str, scratchpad: str, step_number: int, max_steps: int) -> str:
    """
    Builds the per-step "user" message sent alongside the system prompt.

    Extracted from the inline f-string inside `Agent.run` in the notebook.
    This is what tells the model the original query, everything observed so
    far (the scratchpad), and which step it's currently on - the same three
    pieces of context the notebook injected at every loop iteration.

    Args:
        query: the original user question for this run.
        scratchpad: the running Thought/Action/Observation transcript built
                    up over previous steps (empty string on step 1).
        step_number: zero-indexed current step (matches the notebook's loop
                     variable before the +1 adjustment).
        max_steps: the configured maximum number of steps for this run.

    Returns:
        The fully formatted message string to send as the "user" role.
    """
    return (
        f"User Query: {query}\n\n{scratchpad}\n\n"
        f"(You are on step {step_number + 1} of {max_steps} maximum steps.)\n\n"
        "What is your next step?"
    )
