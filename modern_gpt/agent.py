"""A minimal ReAct-style tool-calling agent.

The harness is model-agnostic: it drives any callable `generate_fn(prompt) -> str`
through a Thought -> Action -> Observation loop until the model emits a final
answer or the step budget is exhausted. Tools are registered with a name and a
Python callable.

    Thought: I should add the numbers
    Action: calc[2 + 3]
    Observation: 5
    Answer: 5

This is the standard agent control flow (parse -> dispatch tool -> feed result
back). Fine-tuning the base model on synthetic traces of exactly this format
(see scripts/) teaches it to emit valid actions; the loop itself is what's shown
and tested here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.DOTALL)
ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.DOTALL)


def calc_tool(expr: str) -> str:
    """A safe arithmetic calculator (no names/builtins exposed)."""
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


@dataclass
class Agent:
    generate_fn: object                                  # callable: prompt -> completion
    tools: dict = field(default_factory=lambda: {"calc": calc_tool})
    max_steps: int = 5
    system: str = (
        "You are a tool-using assistant. Use Action: tool[input] to call a tool, "
        "read the Observation, then give Answer: <result>.\n"
    )

    def run(self, question: str):
        transcript = self.system + f"Question: {question}\n"
        trace = []
        for _ in range(self.max_steps):
            completion = self.generate_fn(transcript)
            transcript += completion

            answer = ANSWER_RE.search(completion)
            if answer:
                trace.append(("answer", answer.group(1).strip()))
                return answer.group(1).strip(), trace

            action = ACTION_RE.search(completion)
            if not action:
                trace.append(("halt", "no action or answer produced"))
                break
            tool, arg = action.group(1), action.group(2).strip()
            tool_fn = self.tools.get(tool)
            obs = tool_fn(arg) if tool_fn is not None else f"unknown tool: {tool}"
            trace.append((tool, arg, obs))
            transcript += f"\nObservation: {obs}\n"
        return None, trace
