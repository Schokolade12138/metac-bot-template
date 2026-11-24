import math
import unittest

from bot.guards import clamp_probability, floor_option_probabilities


class ClampProbabilityTest(unittest.TestCase):
    def test_extremes_are_pulled_in(self):
        self.assertEqual(clamp_probability(0.0), 0.02)
        self.assertEqual(clamp_probability(0.999), 0.98)

    def test_middle_values_are_untouched(self):
        self.assertEqual(clamp_probability(0.37), 0.37)

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            clamp_probability(float("nan"))


class FloorOptionProbabilitiesTest(unittest.TestCase):
    def assert_valid(self, result, floor=0.01):
        self.assertTrue(math.isclose(sum(result), 1.0, abs_tol=1e-9))
        self.assertTrue(all(p >= floor - 1e-9 for p in result))

    def test_zero_option_gets_the_floor(self):
        result = floor_option_probabilities([0.0, 0.3, 0.7])
        self.assert_valid(result)
        self.assertAlmostEqual(result[0], 0.01)
        self.assertAlmostEqual(result[1] / result[2], 0.3 / 0.7)

    def test_already_safe_distribution_is_unchanged(self):
        result = floor_option_probabilities([0.2, 0.3, 0.5])
        for got, want in zip(result, [0.2, 0.3, 0.5]):
            self.assertAlmostEqual(got, want)

    def test_option_just_above_floor_does_not_fall_below_it(self):
        result = floor_option_probabilities([0.0, 0.0, 0.0101, 0.9899])
        self.assert_valid(result)

    def test_unnormalised_input_is_rescaled(self):
        self.assert_valid(floor_option_probabilities([1, 1, 2]))

    def test_all_zero_becomes_uniform(self):
        self.assertEqual(floor_option_probabilities([0, 0]), [0.5, 0.5])

    def test_many_options_cap_the_floor(self):
        result = floor_option_probabilities([0.0] * 199 + [1.0])
        self.assert_valid(result, floor=1 / 200)

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            floor_option_probabilities([-0.1, 1.1])


if __name__ == "__main__":
    unittest.main()
