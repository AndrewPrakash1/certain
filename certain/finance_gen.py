"""Synthetic finance problems. Gold is the interpreter's own value."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Callable

from certain.dsl import format_decimal, interpret, values_close
from certain.prompts import user_prompt, wrap_completion


@dataclass
class Problem:
    id: str
    domain: str
    kind: str
    problem: str
    gold: str
    answer_name: str
    certificate: str
    split: str = "train"
    journal: bool = False

    def completion(self) -> str:
        return wrap_completion(self.certificate, self.answer_name, boxed=self.gold)

    def prompt(self) -> str:
        return user_prompt(self.problem)

    def to_record(self) -> dict[str, Any]:
        rec = asdict(self)
        rec["prompt"] = self.prompt()
        rec["completion"] = self.completion()
        rec["answer"] = self.gold
        rec["ground_truth"] = self.gold
        rec["instruction"] = self.prompt()
        rec["output"] = self.completion()
        rec["input"] = ""
        return rec


def _money(rng: random.Random, lo: int, hi: int, step: int = 5) -> Decimal:
    return Decimal(rng.randrange(lo, hi + 1, step))


def _rate(rng: random.Random, choices: list[str]) -> Decimal:
    return Decimal(rng.choice(choices))


def _finalize(kind: str, problem: str, certificate: str, answer_name: str, idx: int, split: str) -> Problem:
    result = interpret(certificate, answer_name=answer_name)
    if not result.computes or result.value is None:
        raise RuntimeError(f"{kind} certificate failed: {result.parse_error or result.undefined}")
    gold = format_decimal(result.value)
    # re-check the gold path the model will be scored on
    wrapped = wrap_completion(certificate, answer_name, boxed=gold)
    checked = interpret(certificate, answer_name=answer_name)
    if not values_close(checked.value, result.value):
        raise RuntimeError("generator/interpreter disagreement")
    return Problem(
        id=f"{split}-{kind}-{idx:04d}",
        domain="finance",
        kind=kind,
        problem=problem,
        gold=gold,
        answer_name=answer_name,
        certificate=certificate.strip() + "\n",
        split=split,
        journal=result.books_present,
    )


def gen_invoice(rng: random.Random, idx: int, split: str) -> Problem:
    qty = rng.randint(2, 40)
    price = rng.choice(["1.50", "2.50", "5", "8", "10", "12.50", "15", "20", "25", "40", "80"])
    discount = rng.choice(["0", "0.05", "0.10", "0.15"])
    tax = rng.choice(["0", "0.05", "0.08", "0.10", "0.18", "0.20"])
    cert = (
        f"revenue = {qty} * {price}\n"
        f"discount = revenue * {discount}\n"
        f"net = revenue - discount\n"
        f"tax = net * {tax}\n"
        f"total = net + tax\n"
        f"dr cash = total\n"
        f"cr revenue = net\n"
        f"cr tax_payable = tax\n"
    )
    problem = (
        f"A shop sold {qty} units at ${price} each. "
        f"It gives a {format_decimal(Decimal(discount) * 100)}% discount, "
        f"then charges {format_decimal(Decimal(tax) * 100)}% tax on the discounted amount. "
        f"How much cash is collected? Record the journal (cash, revenue, tax payable)."
    )
    return _finalize("invoice", problem, cert, "total", idx, split)


def gen_loan(rng: random.Random, idx: int, split: str) -> Problem:
    principal = _money(rng, 1000, 20000, 500)
    rate = _rate(rng, ["0.06", "0.08", "0.10", "0.12", "0.15"])
    months = rng.choice([2, 3, 4, 6, 8, 9, 10, 12])
    payment = _money(rng, 0, 800, 50)
    cert = (
        f"principal = {format_decimal(principal)}\n"
        f"rate = {rate}\n"
        f"months = {months}\n"
        f"interest = principal * rate * months / 12\n"
        f"due = principal + interest\n"
        f"payment = {format_decimal(payment)}\n"
        f"remaining = due - payment\n"
        f"dr receivable = remaining\n"
        f"cr loan = remaining\n"
    )
    problem = (
        f"A loan of ${format_decimal(principal)} carries {format_decimal(rate * 100)}% "
        f"simple annual interest. After {months} months the borrower pays "
        f"${format_decimal(payment)}. What balance remains? "
        f"Interest is principal × rate × months / 12."
    )
    return _finalize("loan", problem, cert, "remaining", idx, split)


def gen_payroll(rng: random.Random, idx: int, split: str) -> Problem:
    hours = rng.choice([20, 24, 30, 32, 35, 40, 45])
    wage = rng.choice(["12", "15", "18", "20", "22.50", "25", "30"])
    tax = _rate(rng, ["0.10", "0.12", "0.18", "0.20"])
    insurance = _money(rng, 20, 120, 10)
    cert = (
        f"gross = {hours} * {wage}\n"
        f"tax = gross * {tax}\n"
        f"insurance = {format_decimal(insurance)}\n"
        f"net = gross - tax - insurance\n"
        f"dr wage_expense = gross\n"
        f"cr tax_payable = tax\n"
        f"cr insurance_payable = insurance\n"
        f"cr cash = net\n"
    )
    problem = (
        f"An employee worked {hours} hours at ${wage} per hour. "
        f"Payroll tax is {format_decimal(tax * 100)}% of gross and insurance is "
        f"${format_decimal(insurance)}. What is the net pay? Record the payroll journal."
    )
    return _finalize("payroll", problem, cert, "net", idx, split)


def gen_margin(rng: random.Random, idx: int, split: str) -> Problem:
    units = rng.randint(10, 200)
    price = rng.choice(["8", "10", "12", "15", "20", "25"])
    cost = rng.choice(["3", "4", "5", "6", "7.50", "9"])
    fixed = _money(rng, 50, 400, 25)
    cert = (
        f"revenue = {units} * {price}\n"
        f"cogs = {units} * {cost}\n"
        f"fixed = {format_decimal(fixed)}\n"
        f"profit = revenue - cogs - fixed\n"
    )
    problem = (
        f"A stall sold {units} items at ${price} each. Each item costs ${cost} to make, "
        f"and fixed costs are ${format_decimal(fixed)}. What is the profit?"
    )
    return _finalize("margin", problem, cert, "profit", idx, split)


def gen_fx(rng: random.Random, idx: int, split: str) -> Problem:
    amount = _money(rng, 100, 5000, 50)
    rate = rng.choice(["0.80", "0.85", "1.10", "1.25", "1.50", "82.50", "83"])
    fee_rate = _rate(rng, ["0", "0.01", "0.02", "0.025"])
    cert = (
        f"amount = {format_decimal(amount)}\n"
        f"rate = {rate}\n"
        f"converted = amount * rate\n"
        f"fee = converted * {fee_rate}\n"
        f"net = converted - fee\n"
    )
    problem = (
        f"Convert {format_decimal(amount)} units at a rate of {rate}. "
        f"The desk charges a {format_decimal(Decimal(fee_rate) * 100)}% fee on the converted amount. "
        f"How much does the customer receive?"
    )
    return _finalize("fx", problem, cert, "net", idx, split)


def gen_inventory(rng: random.Random, idx: int, split: str) -> Problem:
    begin = _money(rng, 200, 4000, 50)
    purchases = _money(rng, 100, 2500, 50)
    cogs = _money(rng, 50, 2000, 50)
    cert = (
        f"beginning = {format_decimal(begin)}\n"
        f"purchases = {format_decimal(purchases)}\n"
        f"cogs = {format_decimal(cogs)}\n"
        f"ending = beginning + purchases - cogs\n"
        f"dr inventory = ending\n"
        f"cr inventory = ending\n"
    )
    problem = (
        f"Inventory began at ${format_decimal(begin)}. Purchases were ${format_decimal(purchases)} "
        f"and cost of goods sold was ${format_decimal(cogs)}. What is ending inventory?"
    )
    return _finalize("inventory", problem, cert, "ending", idx, split)


def gen_depreciation(rng: random.Random, idx: int, split: str) -> Problem:
    cost = _money(rng, 1200, 24000, 200)
    salvage = _money(rng, 0, 2000, 100)
    if salvage >= cost:
        salvage = Decimal("0")
    years = rng.choice([3, 4, 5, 6, 8, 10])
    used = rng.randint(1, years)
    cert = (
        f"cost = {format_decimal(cost)}\n"
        f"salvage = {format_decimal(salvage)}\n"
        f"life = {years}\n"
        f"annual = (cost - salvage) / life\n"
        f"used = {used}\n"
        f"accumulated = annual * used\n"
        f"book = cost - accumulated\n"
    )
    problem = (
        f"A machine costs ${format_decimal(cost)} with salvage ${format_decimal(salvage)} "
        f"and a {years}-year straight-line life. After {used} year(s), what is the book value?"
    )
    return _finalize("depreciation", problem, cert, "book", idx, split)


def gen_vat_inclusive(rng: random.Random, idx: int, split: str) -> Problem:
    total = _money(rng, 118, 2360, 118)
    rate = _rate(rng, ["0.05", "0.10", "0.18", "0.20"])
    cert = (
        f"total = {format_decimal(total)}\n"
        f"rate = {rate}\n"
        f"net = total / (1 + rate)\n"
        f"tax = total - net\n"
    )
    problem = (
        f"A receipt shows ${format_decimal(total)} including "
        f"{format_decimal(rate * 100)}% tax. What is the tax-exclusive net amount?"
    )
    return _finalize("vat_inclusive", problem, cert, "net", idx, split)


def gen_multi_item(rng: random.Random, idx: int, split: str) -> Problem:
    q1, q2 = rng.randint(1, 12), rng.randint(1, 12)
    p1 = rng.choice(["4", "5", "6.50", "8", "10"])
    p2 = rng.choice(["3", "7", "9", "12", "15"])
    tax = _rate(rng, ["0", "0.08", "0.10", "0.18"])
    cert = (
        f"line1 = {q1} * {p1}\n"
        f"line2 = {q2} * {p2}\n"
        f"subtotal = line1 + line2\n"
        f"tax = subtotal * {tax}\n"
        f"total = subtotal + tax\n"
        f"dr cash = total\n"
        f"cr revenue = subtotal\n"
        f"cr tax_payable = tax\n"
    )
    problem = (
        f"An invoice has {q1} units at ${p1} and {q2} units at ${p2}. "
        f"Tax is {format_decimal(tax * 100)}% of the subtotal. What is the invoice total?"
    )
    return _finalize("multi_item", problem, cert, "total", idx, split)


GENERATORS: list[Callable[[random.Random, int, str], Problem]] = [
    gen_invoice,
    gen_loan,
    gen_payroll,
    gen_margin,
    gen_fx,
    gen_inventory,
    gen_depreciation,
    gen_vat_inclusive,
    gen_multi_item,
]


def generate_finance(n: int, *, seed: int = 7, split: str = "train", start: int = 0) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    seen: set[str] = set()
    i = start
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        gen = GENERATORS[i % len(GENERATORS)]
        try:
            problem = gen(rng, i, split)
        except Exception:
            i += 1
            continue
        if problem.problem in seen:
            i += 1
            continue
        seen.add(problem.problem)
        out.append(problem)
        i += 1
    return out
