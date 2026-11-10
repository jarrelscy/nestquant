import unittest

import numpy as np

from kld_math import summarize, tokenwise_kld


class TokenwiseKldTest(unittest.TestCase):
    def test_identical_logits_have_zero_divergence(self):
        logits = np.random.default_rng(0).normal(size=(5, 11)).astype(np.float32)
        np.testing.assert_allclose(tokenwise_kld(logits, logits), np.zeros(5), atol=1e-12)

    def test_matches_closed_form_in_teacher_to_student_direction(self):
        teacher = np.log(np.array([[0.5, 0.25, 0.25]]))
        student = np.log(np.array([[0.25, 0.5, 0.25]]))
        expected = 0.5 * np.log(2) + 0.25 * np.log(0.5)
        np.testing.assert_allclose(tokenwise_kld(teacher, student), [expected], rtol=1e-12)
        reverse = 0.25 * np.log(0.5) + 0.5 * np.log(2)
        asymmetric_teacher = np.log(np.array([[0.7, 0.2, 0.1]]))
        forward = tokenwise_kld(asymmetric_teacher, student)[0]
        backward = tokenwise_kld(student, asymmetric_teacher)[0]
        self.assertNotAlmostEqual(forward, backward, places=6)
        self.assertAlmostEqual(expected, reverse)

    def test_is_invariant_to_per_row_logit_offsets(self):
        rng = np.random.default_rng(1)
        teacher = rng.normal(size=(4, 9))
        student = rng.normal(size=(4, 9))
        shifted = tokenwise_kld(teacher + 100.0, student - 37.5)
        np.testing.assert_allclose(shifted, tokenwise_kld(teacher, student), rtol=1e-10)

    def test_remains_finite_for_extreme_float32_logits(self):
        teacher = np.array([[80.0, -80.0, 0.0]], dtype=np.float32)
        student = np.array([[-80.0, 80.0, 0.0]], dtype=np.float32)
        value = tokenwise_kld(teacher, student)
        self.assertTrue(np.isfinite(value).all())
        self.assertGreater(value[0], 100.0)

    def test_rejects_mismatched_or_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            tokenwise_kld(np.zeros((2, 3)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            tokenwise_kld(np.zeros(3), np.zeros(3))
        with self.assertRaises(ValueError):
            tokenwise_kld(np.array([[0.0, np.nan]]), np.zeros((1, 2)))

    def test_summary_reports_mean_quantiles_and_top1_agreement(self):
        teacher = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0]])
        student = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
        report = summarize(tokenwise_kld(teacher, student), teacher, student)
        self.assertEqual(report['count'], 4)
        self.assertAlmostEqual(report['top1_agreement'], 0.75)
        self.assertAlmostEqual(report['mean'], tokenwise_kld(teacher, student).mean())
        self.assertEqual(report['p50'], 0.0)
        self.assertGreater(report['max'], 0.0)


if __name__ == '__main__':
    unittest.main()
