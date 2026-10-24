from __future__ import annotations

import unittest

from certain.curate import gate
from certain.dsl import interpret_text, values_close, parse_number
from certain.finance_gen import generate_finance
from certain.math_gen import generate_math
from certain.rewards import certificate_reward


class TestGenerators(unittest.TestCase):
    def test_finance_certificates_execute(self):
        problems = generate_finance(45, seed=3, split="train")
        self.assertGreaterEqual(len(problems), 40)
        kinds = {p.kind for p in problems}
        self.assertIn("invoice", kinds)
        self.assertIn("loan", kinds)
        for problem in problems:
            result = interpret_text(problem.completion())
            self.assertTrue(result.computes, msg=f"{problem.id}: {result.parse_error}")
            self.assertTrue(values_close(result.value, parse_number(problem.gold)))
            self.assertEqual(certificate_reward(problem.completion(), reference=problem.gold), 1.0)

    def test_math_certificates_execute(self):
        problems = generate_math(20, seed=5, split="heldout")
        self.assertGreaterEqual(len(problems), 15)
        for problem in problems:
            self.assertEqual(problem.domain, "math")
            self.assertEqual(certificate_reward(problem.completion(), reference=problem.gold), 1.0)

    def test_gate_rejects_broken_certificate(self):
        problems = generate_finance(3, seed=1)
        problems[0].certificate = "total = assert True\n"
        kept, rejected = gate(problems)
        self.assertTrue(any(r["reason"] != "duplicate" for r in rejected) or len(kept) < len(problems))


if __name__ == "__main__":
    unittest.main()
