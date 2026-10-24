from __future__ import annotations

import json
import unittest
from pathlib import Path

from certain.metrics import score_completion

CASES = json.loads((Path(__file__).resolve().parents[1] / "demo" / "cases.json").read_text())


class TestDemoCases(unittest.TestCase):
    def _case(self, case_id: str) -> dict:
        return next(c for c in CASES["cases"] if c["id"] == case_id)

    def test_invoice_certain_is_faithful(self):
        c = self._case("invoice")
        s = score_completion(c["models"]["certain"], c["gold"], c["id"])
        self.assertTrue(s.faithful)
        self.assertEqual(s.certificate_reward, 1.0)

    def test_invoice_outcome_is_unfaithful_correct(self):
        c = self._case("invoice")
        s = score_completion(c["models"]["outcome"], c["gold"], c["id"])
        self.assertTrue(s.accuracy)
        self.assertTrue(s.unfaithful_correct)
        self.assertEqual(s.outcome_reward, 1.0)
        self.assertEqual(s.certificate_reward, 0.0)

    def test_loan_base_uses_wrong_formula(self):
        c = self._case("loan")
        s = score_completion(c["models"]["base"], c["gold"], c["id"])
        self.assertFalse(s.accuracy)
        self.assertTrue(s.parsed)

    def test_unbalanced_outcome_pays_and_certain_does_not(self):
        c = self._case("unbalanced")
        outcome = score_completion(c["models"]["outcome"], c["gold"], "u")
        certain = score_completion(c["models"]["certain"], c["gold"], "c")
        self.assertTrue(outcome.accuracy)
        self.assertTrue(outcome.unfaithful_correct)
        self.assertLess(outcome.certificate_reward, 1.0)
        self.assertTrue(certain.faithful)
