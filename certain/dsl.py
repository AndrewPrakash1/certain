"""Closed certificate language. The model cannot write the checker.

Grammar, one statement per line:

    NAME = EXPR
    dr ACCOUNT = EXPR
    cr ACCOUNT = EXPR

EXPR uses numbers, already-bound names, parentheses, and + - * /.
Anything else — calls, imports, indexing, attributes, juxtaposition —
is illegal and the whole certificate scores 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, getcontext
from typing import Iterable

getcontext().prec = 28

TOLERANCE = Decimal("1e-6")

_CERT_RE = re.compile(r"<certificate>(.*?)</certificate>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")

_ILLEGAL_CHARS = set("[]{},;:'\"`@#$\\!&|~?<>")


class _Tok:
    NUMBER = "NUMBER"
    NAME = "NAME"
    OP = "OP"
    END = "END"


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


@dataclass
class LineResult:
    line_no: int
    raw: str
    kind: str
    name: str | None = None
    value: Decimal | None = None
    ok: bool = False
    error: str | None = None


@dataclass
class CertificateResult:
    parsed: bool
    parse_error: str | None = None
    env: dict[str, Decimal] = field(default_factory=dict)
    lines: list[LineResult] = field(default_factory=list)
    undefined: list[str] = field(default_factory=list)
    illegal: list[str] = field(default_factory=list)
    debits: Decimal = Decimal("0")
    credits: Decimal = Decimal("0")
    books_present: bool = False
    books_balanced: bool = True
    answer_name: str | None = None
    value: Decimal | None = None
    boxed_value: Decimal | None = None
    boxed_disagrees: bool = False
    raw_certificate: str = ""

    @property
    def no_undefined(self) -> bool:
        return self.parsed and not self.undefined

    @property
    def books_ok(self) -> bool:
        return self.parsed and (not self.books_present or self.books_balanced)

    @property
    def computes(self) -> bool:
        return (
            self.parsed
            and self.no_undefined
            and self.books_ok
            and not self.boxed_disagrees
            and self.value is not None
        )


@dataclass
class Extracted:
    certificate: str | None
    answer_token: str | None
    boxed_token: str | None
    raw: str


def extract_blocks(text: str) -> Extracted:
    if not text:
        return Extracted(None, None, None, text or "")
    cert_m = _CERT_RE.search(text)
    ans_m = _ANSWER_RE.search(text)
    box_m = _BOXED_RE.search(text)
    return Extracted(
        certificate=cert_m.group(1).strip() if cert_m else None,
        answer_token=ans_m.group(1).strip() if ans_m else None,
        boxed_token=box_m.group(1).strip() if box_m else None,
        raw=text,
    )


def parse_number(token: str) -> Decimal | None:
    token = (token or "").strip().replace(",", "")
    if not token:
        return None
    if token.startswith("$"):
        token = token[1:]
    if not _NUMBER_RE.match(token) and not re.match(r"^-?\d+(?:\.\d+)?$", token):
        return None
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def values_close(a: Decimal | float | int | None, b: Decimal | float | int | None) -> bool:
    if a is None or b is None:
        return False
    da = a if isinstance(a, Decimal) else Decimal(str(a))
    db = b if isinstance(b, Decimal) else Decimal(str(b))
    return abs(da - db) <= TOLERANCE


def _tokenize_expr(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _ILLEGAL_CHARS:
            raise ValueError(f"illegal character {ch!r}")
        if ch in "+-*/()=":
            tokens.append(Token(_Tok.OP, ch, i))
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            lex = src[i:j]
            if lex.count(".") > 1 or not _NUMBER_RE.match(lex):
                raise ValueError(f"illegal number {lex!r}")
            tokens.append(Token(_Tok.NUMBER, lex, i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            lex = src[i:j]
            if j < n and src[j] == ".":
                raise ValueError(f"attribute access on {lex!r}")
            tokens.append(Token(_Tok.NAME, lex, i))
            i = j
            continue
        raise ValueError(f"illegal character {ch!r}")
    tokens.append(Token(_Tok.END, "", n))
    return tokens


class _ExprParser:
    def __init__(self, tokens: list[Token], env: dict[str, Decimal]):
        self.tokens = tokens
        self.pos = 0
        self.env = env
        self.undefined: list[str] = []

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def eat(self, kind: str | None = None, value: str | None = None) -> Token:
        tok = self.peek()
        if kind and tok.kind != kind:
            raise ValueError(f"expected {kind}, got {tok.kind} {tok.value!r}")
        if value is not None and tok.value != value:
            raise ValueError(f"expected {value!r}, got {tok.value!r}")
        self.pos += 1
        return tok

    def parse(self) -> Decimal:
        value = self._expr()
        if self.peek().kind != _Tok.END:
            raise ValueError(f"trailing tokens after expression: {self.peek().value!r}")
        return value

    def _expr(self) -> Decimal:
        value = self._term()
        while self.peek().kind == _Tok.OP and self.peek().value in "+-":
            op = self.eat().value
            rhs = self._term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def _term(self) -> Decimal:
        value = self._factor()
        while self.peek().kind == _Tok.OP and self.peek().value in "*/":
            op = self.eat().value
            rhs = self._factor()
            if op == "*":
                value = value * rhs
            else:
                if rhs == 0:
                    raise ValueError("division by zero")
                value = value / rhs
        return value

    def _factor(self) -> Decimal:
        tok = self.peek()
        if tok.kind == _Tok.OP and tok.value == "+":
            self.eat()
            return self._factor()
        if tok.kind == _Tok.OP and tok.value == "-":
            self.eat()
            return -self._factor()
        if tok.kind == _Tok.OP and tok.value == "(":
            self.eat()
            value = self._expr()
            if self.peek().kind != _Tok.OP or self.peek().value != ")":
                raise ValueError("unbalanced parentheses")
            self.eat()
            nxt = self.peek()
            if nxt.kind in (_Tok.NUMBER, _Tok.NAME) or (nxt.kind == _Tok.OP and nxt.value == "("):
                raise ValueError("juxtaposition is illegal")
            return value
        if tok.kind == _Tok.NUMBER:
            self.eat()
            nxt = self.peek()
            if nxt.kind in (_Tok.NUMBER, _Tok.NAME) or (nxt.kind == _Tok.OP and nxt.value == "("):
                raise ValueError("juxtaposition is illegal")
            return Decimal(tok.value)
        if tok.kind == _Tok.NAME:
            self.eat()
            nxt = self.peek()
            if nxt.kind == _Tok.OP and nxt.value == "(":
                raise ValueError(f"function call {tok.value}(...) is illegal")
            if nxt.kind in (_Tok.NUMBER, _Tok.NAME):
                raise ValueError("juxtaposition is illegal")
            if tok.value not in self.env:
                self.undefined.append(tok.value)
                raise ValueError(f"undefined: {tok.value}")
            return self.env[tok.value]
        raise ValueError(f"unexpected token {tok.value!r}")


def _split_assignment(raw: str) -> tuple[str, str] | None:
    if "=" not in raw:
        return None
    left, right = raw.split("=", 1)
    return left.strip(), right.strip()


def interpret(certificate: str, answer_name: str | None = None) -> CertificateResult:
    """Execute a certificate body (no wrapper tags)."""
    result = CertificateResult(parsed=False, raw_certificate=certificate or "", answer_name=answer_name)
    if certificate is None:
        result.parse_error = "missing certificate"
        return result

    env: dict[str, Decimal] = {}
    lines: list[LineResult] = []
    illegal: list[str] = []
    undefined: list[str] = []
    debits = Decimal("0")
    credits = Decimal("0")
    books_present = False
    parsed_ok = True

    for idx, raw in enumerate((certificate or "").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(LineResult(idx, raw, "blank", ok=True))
            continue

        lower = stripped.lower()
        kind = "assign"
        name_part = stripped
        if lower.startswith("dr ") or lower.startswith("cr "):
            kind = "debit" if lower.startswith("dr ") else "credit"
            name_part = stripped[3:].strip()
            books_present = True

        split = _split_assignment(name_part)
        if split is None:
            parsed_ok = False
            msg = "expected NAME = EXPR"
            illegal.append(msg)
            lines.append(LineResult(idx, raw, "error", ok=False, error=msg))
            continue

        lhs, rhs = split
        if not _NAME_RE.match(lhs):
            parsed_ok = False
            msg = f"illegal name {lhs!r}"
            illegal.append(msg)
            lines.append(LineResult(idx, raw, "error", name=lhs, ok=False, error=msg))
            continue
        if not rhs:
            parsed_ok = False
            msg = "empty expression"
            illegal.append(msg)
            lines.append(LineResult(idx, raw, "error", name=lhs, ok=False, error=msg))
            continue

        try:
            tokens = _tokenize_expr(rhs)
            parser = _ExprParser(tokens, env)
            value = parser.parse()
            undefined.extend(parser.undefined)
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("undefined:"):
                undefined.append(msg.split(":", 1)[1].strip())
                lines.append(LineResult(idx, raw, kind, name=lhs, ok=False, error=msg))
            else:
                parsed_ok = False
                illegal.append(msg)
                lines.append(LineResult(idx, raw, "error", name=lhs, ok=False, error=msg))
            continue

        env[lhs] = value
        if kind == "debit":
            debits += value
        elif kind == "credit":
            credits += value
        lines.append(LineResult(idx, raw, kind, name=lhs, value=value, ok=True))

    books_balanced = (not books_present) or values_close(debits, credits)
    answer_value = env.get(answer_name) if answer_name and _NAME_RE.match(answer_name) else None

    result.parsed = parsed_ok and not illegal
    result.parse_error = illegal[0] if illegal else None
    result.env = env
    result.lines = lines
    result.undefined = list(dict.fromkeys(undefined))
    result.illegal = list(dict.fromkeys(illegal))
    result.debits = debits
    result.credits = credits
    result.books_present = books_present
    result.books_balanced = books_balanced
    result.value = answer_value
    return result


def interpret_text(text: str) -> CertificateResult:
    """Interpret a model completion that may wrap a certificate."""
    extracted = extract_blocks(text)
    answer_name = None
    boxed_value = parse_number(extracted.boxed_token) if extracted.boxed_token else None
    if extracted.answer_token:
        token = extracted.answer_token.strip()
        if _NAME_RE.match(token):
            answer_name = token
        else:
            maybe = parse_number(token)
            if maybe is not None:
                boxed_value = boxed_value if boxed_value is not None else maybe

    if extracted.certificate is None:
        result = CertificateResult(
            parsed=False,
            parse_error="missing <certificate> block",
            answer_name=answer_name,
            boxed_value=boxed_value,
        )
        return result

    result = interpret(extracted.certificate, answer_name=answer_name)
    result.answer_name = answer_name
    result.boxed_value = boxed_value
    if boxed_value is not None and result.value is not None:
        result.boxed_disagrees = not values_close(boxed_value, result.value)
    elif boxed_value is not None and result.value is None and answer_name:
        result.boxed_disagrees = True
    return result


def format_decimal(value: Decimal | float | int) -> str:
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def iter_nonempty_lines(certificate: str) -> Iterable[str]:
    for line in certificate.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield stripped
