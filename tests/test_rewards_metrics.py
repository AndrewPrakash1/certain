from __future__ import annotations

import unittest

from certain.metrics import aggregate, score_completion
from certain.rewards import group_advantages, outcome_reward
from certain.prompts import wrap_completion


class TestMetrics(unittest.TestCase):
    def test_faithful_vs_unfaithful(self):
        gold = "127.44"
        faithful = wrap_completion(
            "revenue = 80 * 1.5\ndiscount = revenue * 0.1\nnet = revenue - discount\n"
            "tax = net * 0.18\ntotal = net + tax\n",
            "total",
            boxed=gold,
        )
        unfaithful = "I multiplied wrongly but the total is 127.44\n\\boxed{127.44}"
        s1 = score_completion(faithful, gold, "f")
        s2 = score_completion(unfaithful, gold, "u")
        self.assertTrue(s1.faithful)
        self.assertFalse(s1.unfaithful_correct)
        self.assertTrue(s2.accuracy)
        self.assertTrue(s2.unfaithful_correct)
        report = aggregate([s1, s2])
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertAlmostEqual(report.faithful_accuracy, 0.5)
        self.assertAlmostEqual(report.unfaithful_correct_rate, 0.5)

    def test_outcome_binary(self):
        self.assertEqual(outcome_reward("\\boxed{4}", reference=4), 1.0)
        self.assertEqual(outcome_reward("\\boxed{5}", reference=4), 0.0)

    def test_group_advantages_zero_var(self):
        self.assertEqual(group_advantages([1, 1, 1, 1]), [0.0, 0.0, 0.0, 0.0])

    def test_group_advantages_contrast(self):
        adv = group_advantages([0.0, 1.0, 0.0, 1.0])
        self.assertGreater(adv[1], 0)
        self.assertLess(adv[0], 0)


if __name__ == "__main__":
    unittest.main()
