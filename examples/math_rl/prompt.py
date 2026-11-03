"""Task prompt shared by dataset projection and the harness loop."""

SOLUTION_PATH = "/workspace/solution.md"
SCRATCH_DIR = "/workspace/scratch"

_PROMPT_TEMPLATE = """Solve the following research-level mathematics problem.

Problem:
{problem}

Requirements:
- Give a rigorous, self-contained solution. State every lemma you use and prove it.
- You may use the sandbox to test examples, run computations, or check special cases.
  Put scratch files under {scratch_dir}.
- Do not use web search or subagents.
- Write the final solution to {solution_path}.
- Use Markdown. Put the main theorem or final answer last.
- If the problem asks for a counterexample, give an explicit construction and verify it.
- If the problem is computational, include the code and the numerical checks you ran.
"""


def build_task_prompt(problem: str) -> str:
    """
    Render one problem as the harness user prompt.

    Args:
        problem (str): The problem statement.

    Returns:
        str: The complete user prompt.
    """
    statement = problem.strip()
    if not statement:
        raise ValueError("Problem statement cannot be empty.")
    return _PROMPT_TEMPLATE.format(
        problem=statement,
        solution_path=SOLUTION_PATH,
        scratch_dir=SCRATCH_DIR,
    )


def build_messages(problem: str) -> list[dict]:
    """
    Build the PSRL message list for one problem.

    Args:
        problem (str): The problem statement.

    Returns:
        list[dict]: One user message in OpenAI format.
    """
    return [{"role": "user", "content": build_task_prompt(problem)}]
