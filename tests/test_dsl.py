"""Interpreter: legal programs run, illegal programs score as illegal."""

from __future__ import annotations

import unittest

from certain.dsl import interpret, interpret_text, values_close
from certain.rewards import certificate_reward, outcome_reward


PLAN_CERT = """
revenue = 80 * 1.5
discount = revenue * 0.1
net = revenue - discount
tax = net * 0.18
total = net + tax
dr cash = total
cr revenue = net
cr tax_payable = tax
"""


class TestPlanExample(unittest.TestCase):
    def test_plan_certificate_computes_total(self):
        result = interpret(PLAN_CERT, answer_name="total")
        self.assertTrue(result.parsed)
        self.assertTrue(result.no_undefined)
        self.assertTrue(result.books_balanced)
        self.assertTrue(values_close(result.value, 127.44))


class TestIllegalRejected(unittest.TestCase):
    def test_assert_true_words(self):
        result = interpret("ok = assert True\n", answer_name="ok")
        self.assertFalse(result.parsed)
        self.assertTrue(result.illegal)

    def test_assert_true_call(self):
        result = interpret("ok = assert(True)\n", answer_name="ok")
        self.assertFalse(result.parsed)
        self.assertTrue(any("call" in x or "illegal" in x or "juxtaposition" in x for x in result.illegal + [result.parse_error or ""]))

    def test_import(self):
        result = interpret("os = import os\n", answer_name="os")
        self.assertFalse(result.parsed)

    def test_attribute_access(self):
        result = interpret("x = 1\ny = x.__class__\n", answer_name="y")
        self.assertFalse(result.parsed)

    def test_indexing(self):
        result = interpret("x = [1]\n", answer_name="x")
        self.assertFalse(result.parsed)

    def test_print_call(self):
        result = interpret("x = print(1)\n", answer_name="x")
        self.assertFalse(result.parsed)

    def test_reward_is_zero_on_assert(self):
        text = "<certificate>\nok = assert True\n</certificate>\n<answer>ok</answer>"
        self.assertEqual(certificate_reward(text, reference=1), 0.0)


class TestUndefinedAndBooks(unittest.TestCase):
    def test_undefined_name(self):
        result = interpret("total = net + 1\n", answer_name="total")
        self.assertTrue(result.parsed)
        self.assertIn("net", result.undefined)
        self.assertFalse(result.no_undefined)

    def test_unbalanced_journal(self):
        cert = "x = 100\ndr cash = 120\ncr revenue = 100\n"
        result = interpret(cert, answer_name="x")
        self.assertTrue(result.parsed)
        self.assertTrue(result.books_present)
        self.assertFalse(result.books_balanced)
        text = f"<certificate>\n{cert}</certificate>\n<answer>x</answer>"
        # parses + defined + no match (books fail so no 0.4, no 0.2 books)
        self.assertAlmostEqual(certificate_reward(text, reference=100), 0.4)

    def test_balanced_journal_reward(self):
        text = (
            "<certificate>\n"
            f"{PLAN_CERT}</certificate>\n<answer>total</answer>"
        )
        self.assertAlmostEqual(certificate_reward(text, reference=127.44), 1.0)


class TestAnswerAndBoxed(unittest.TestCase):
    def test_missing_certificate(self):
        result = interpret_text("the answer is 4")
        self.assertFalse(result.parsed)

    def test_boxed_disagrees(self):
        text = (
            "<certificate>\nx = 4\n</certificate>\n<answer>x</answer>\n\\boxed{7}"
        )
        result = interpret_text(text)
        self.assertTrue(result.boxed_disagrees)
        self.assertAlmostEqual(certificate_reward(text, reference=4), 0.6)

    def test_outcome_ignores_certificate(self):
        text = "<certificate>\nx = 1 + 1\n</certificate>\n<answer>x</answer>\n\\boxed{118.8}"
        self.assertEqual(outcome_reward(text, reference=118.8), 1.0)
        self.assertAlmostEqual(certificate_reward(text, reference=118.8), 0.6)


class TestExpressions(unittest.TestCase):
    def test_parens_and_unary(self):
        result = interpret("x = (3 + 5) * -2\n", answer_name="x")
        self.assertTrue(result.parsed)
        self.assertTrue(values_close(result.value, -16))

    def test_division_by_zero(self):
        result = interpret("x = 1 / 0\n", answer_name="x")
        self.assertFalse(result.parsed)


if __name__ == "__main__":
    unittest.main()
