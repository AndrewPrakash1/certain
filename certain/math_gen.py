"""GSM8K-style word problems in the same certificate language."""

from __future__ import annotations

import random
from decimal import Decimal

from certain.dsl import format_decimal, interpret
from certain.finance_gen import Problem, _finalize
from certain.prompts import wrap_completion


def gen_bakery(rng: random.Random, idx: int, split: str) -> Problem:
    boxes = rng.randint(4, 20)
    each = rng.choice([4, 6, 8, 10, 12])
    leftover = rng.randint(0, each - 1)
    more = rng.randint(2, 15)
    cert = (
        f"packed = {boxes} * {each}\n"
        f"leftover = {leftover}\n"
        f"baked = packed + leftover\n"
        f"extra = {more}\n"
        f"total = baked + extra\n"
    )
    problem = (
        f"A bakery packed {boxes} boxes with {each} muffins each and had {leftover} muffins left. "
        f"It then baked {more} more. How many muffins are there in all?"
    )
    return _finalize("bakery", problem, cert, "total", idx, split)


def gen_ages(rng: random.Random, idx: int, split: str) -> Problem:
    younger = rng.randint(6, 16)
    diff = rng.randint(3, 12)
    years = rng.randint(2, 8)
    cert = (
        f"younger = {younger}\n"
        f"older = younger + {diff}\n"
        f"later_older = older + {years}\n"
        f"later_younger = younger + {years}\n"
        f"sum = later_older + later_younger\n"
    )
    problem = (
        f"Rina is {younger} years old. Her brother is {diff} years older. "
        f"What will the sum of their ages be in {years} years?"
    )
    return _finalize("ages", problem, cert, "sum", idx, split)


def gen_split_money(rng: random.Random, idx: int, split: str) -> Problem:
    people = rng.choice([3, 4, 5, 6])
    share = rng.choice([8, 10, 12, 15, 20, 25])
    extra = rng.choice([2, 4, 5, 6, 10])
    cert = (
        f"people = {people}\n"
        f"share = {share}\n"
        f"base = people * share\n"
        f"extra = {extra}\n"
        f"total = base + extra\n"
    )
    problem = (
        f"{people} friends each paid ${share} for a trip, then they paid ${extra} more for snacks. "
        f"How much did they spend altogether?"
    )
    return _finalize("split_money", problem, cert, "total", idx, split)


def gen_remaining(rng: random.Random, idx: int, split: str) -> Problem:
    start = rng.randrange(40, 200, 5)
    first = rng.randrange(5, 30, 5)
    second = rng.randrange(5, 25, 5)
    cert = (
        f"start = {start}\n"
        f"first = {first}\n"
        f"second = {second}\n"
        f"left = start - first - second\n"
    )
    problem = (
        f"A stall began the day with {start} apples. It sold {first} in the morning "
        f"and {second} in the afternoon. How many apples remain?"
    )
    return _finalize("remaining", problem, cert, "left", idx, split)


def gen_ratio(rng: random.Random, idx: int, split: str) -> Problem:
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    units = rng.choice([4, 5, 6, 8, 10])
    cert = (
        f"parts_a = {a}\n"
        f"parts_b = {b}\n"
        f"unit = {units}\n"
        f"share_a = parts_a * unit\n"
        f"share_b = parts_b * unit\n"
        f"total = share_a + share_b\n"
    )
    problem = (
        f"A prize is split in the ratio {a}:{b}. Each part is worth {units}. "
        f"What is the total prize?"
    )
    return _finalize("ratio", problem, cert, "total", idx, split)


MATH_GENERATORS = [gen_bakery, gen_ages, gen_split_money, gen_remaining, gen_ratio]


def generate_math(n: int, *, seed: int = 11, split: str = "train", start: int = 0) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    seen: set[str] = set()
    i = start
    while len(out) < n and i < start + n * 20:
        gen = MATH_GENERATORS[i % len(MATH_GENERATORS)]
        try:
            problem = gen(rng, i, split)
        except Exception:
            i += 1
            continue
        problem.domain = "math"
        if problem.problem in seen:
            i += 1
            continue
        seen.add(problem.problem)
        out.append(problem)
        i += 1
    return out


def certificate_from_gsm8k_solution(question: str, solution: str, gold: str) -> str | None:
    """Best-effort translation of a GSM8K scratchpad into the closed language.

    GSM8K solutions interleave words and <<a-b=c>> calculations. We keep a
    sample only when those calculations re-execute to the official gold.
    """
    import re

    calc_re = re.compile(r"<<([^=]+)=([^>]+)>>")
    assigns: list[str] = []
    env_ok = True
    for i, (expr, result) in enumerate(calc_re.findall(solution), start=1):
        expr = expr.strip().replace(" ", "")
        if any(ch in expr for ch in "[]{},;:'\"`@#!&|~?<>"):
            env_ok = False
            break
        # rewrite to NAME = EXPR using the official intermediate
        name = f"s{i}"
        # only allow digits, operators, parentheses, dots
        if not re.fullmatch(r"[0-9+\-*/().]+", expr):
            env_ok = False
            break
        assigns.append(f"{name} = {expr}")
    if not assigns or not env_ok:
        return None
    last = f"s{len(assigns)}"
    cert = "\n".join(assigns) + "\n"
    try:
        result = interpret(cert, answer_name=last)
    except Exception:
        return None
    if result.value is None:
        return None
    from certain.dsl import parse_number, values_close

    gold_d = parse_number(gold)
    if gold_d is None or not values_close(result.value, gold_d):
        return None
    return wrap_completion(cert, last, boxed=format_decimal(result.value))
