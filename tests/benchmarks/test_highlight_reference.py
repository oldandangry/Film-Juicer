"""Independent endpoint and boundary controls for the experiment oracle."""

from decimal import Decimal, localcontext
import math
import unittest

import highlight_reference as reference


class HighlightReferenceTests(unittest.TestCase):
    def test_onset_and_peak_use_exact_descriptor_values(self):
        low = 1.0
        high = reference.next_float(low)
        for protection in (math.nextafter(low, 0), low, (low + high) * 0.5, math.nextafter(high, 0)):
            row = reference._row("endpoint", protection, 28.0, 16.0, high, values=[low, high])
            self.assertEqual(row["status"], "Applied")
            for word, result in zip(row["input_bits"], row["expected"], strict=True):
                sample = reference.from_bits(word)
                actual = Decimal(result)
                if sample <= protection:
                    self.assertEqual(actual, Decimal.from_float(sample))
                if sample == high:
                    self.assertEqual(actual, Decimal.from_float(high) * 16)

    def test_overflow_boundary_is_not_rounded_to_float_gain(self):
        gain = math.exp2(0.125)
        maximum = reference.rounded(reference.from_bits(0x7F7FFFFF) / gain)
        while maximum * gain > reference.from_bits(0x7F7FFFFF):
            maximum = reference.next_float(maximum, -1)
        valid = reference._row("valid", 0.184, 1.0, gain, maximum, values=[0.0, maximum])
        invalid_maximum = reference.next_float(maximum)
        invalid = reference._row("invalid", 0.184, 1.0, gain, invalid_maximum, values=[0.0, invalid_maximum])
        self.assertEqual(valid["status"], "Applied")
        self.assertEqual(invalid["status"], "PeakOverflow")

    def test_multiblock_rows_place_peak_and_invalid_in_last_block(self):
        rows = {row["id"]: row for row in reference.generate()["rows"]}
        peak = rows["supplement-multiblock-peak"]
        invalid = rows["supplement-multiblock-invalid"]
        self.assertEqual((peak["width"], peak["height"], peak["status"]), (513, 2, "Applied"))
        self.assertEqual(peak["input_bits"][-1], peak["maximum_bits"])
        self.assertNotIn(peak["maximum_bits"], peak["input_bits"][:-1])
        self.assertEqual(invalid["status"], "InvalidCarrier")
        self.assertTrue(math.isnan(reference.from_bits(invalid["input_bits"][-2])))
        self.assertEqual(len(invalid["input_bits"]), 3 * 513 * 2)

    def test_analytic_continuation_is_positive_at_tiny_argument(self):
        with localcontext() as context:
            context.prec = 80
            self.assertEqual(reference._normalized(Decimal(0)), Decimal("0.5"))
            tiny = Decimal("1e-30")
            actual = reference._normalized(tiny)
            self.assertGreater(actual, Decimal("0.5"))
            self.assertLess(abs(actual - Decimal("0.5") - tiny / 6), Decimal("1e-60"))


if __name__ == "__main__":
    unittest.main()
