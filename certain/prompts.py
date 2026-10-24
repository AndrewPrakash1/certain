"""Shared system / user prompts. Both GRPO runs use this exact text."""

SYSTEM_PROMPT = """You are a careful accounting and math assistant.
Solve every problem by writing a CERTAIN certificate the interpreter can re-run.

Rules:
- Write only assignments `NAME = EXPR` and optional journal lines `dr ACCOUNT = EXPR` or `cr ACCOUNT = EXPR`.
- EXPR may use numbers, names already defined above, parentheses, and + - * /.
- No function calls, imports, indexing, attributes, or extra words inside the certificate.
- If you record a journal, debits must equal credits.
- The final answer is the value of a name you assigned. Put that name in <answer>.

Respond in this exact shape:

<certificate>
revenue = 80 * 1.5
discount = revenue * 0.1
net = revenue - discount
tax = net * 0.18
total = net + tax
dr cash = total
cr revenue = net
cr tax_payable = tax
</certificate>
<answer>total</answer>
"""


def user_prompt(problem: str) -> str:
    return f"Problem:\n{problem.strip()}\n\nWrite a CERTAIN certificate, then name the answer variable."


def wrap_completion(certificate: str, answer_name: str, boxed: str | None = None) -> str:
    body = certificate.strip()
    text = f"<certificate>\n{body}\n</certificate>\n<answer>{answer_name}</answer>"
    if boxed is not None:
        text += f"\n\\boxed{{{boxed}}}"
    return text
