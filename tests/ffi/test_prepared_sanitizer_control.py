from __future__ import annotations

import unittest

from run_prepared_sanitizer_control import verify_failure


class SanitizerControlTests(unittest.TestCase):
    def test_memory_requires_the_production_decoder_and_address_report(self):
        verify_failure("memory", 1, "AddressSanitizer: heap-use-after-free\ndecode_prepared_descriptors")
        outputs = ("", "segmentation fault", "AddressSanitizer: heap-use-after-free",
                   "decode_prepared_descriptors", "AddressSanitizer: heap-buffer-overflow\ndecode_prepared_descriptors")
        self.subtest_cases = ("empty", "signal-only", "address-report-only", "decoder-only", "wrong-address-report")
        for case_id, output in zip(self.subtest_cases, outputs):
            with self.subTest(case=case_id), self.assertRaises(ValueError):
                verify_failure("memory", 1, output)

    def test_undefined_requires_signed_overflow_report(self):
        verify_failure("undefined", 1, "runtime error: signed integer overflow")
        with self.assertRaises(ValueError):
            verify_failure("undefined", 1, "UndefinedBehaviorSanitizer: other error")

    def test_success_never_qualifies_a_control(self):
        controls = (("memory", "AddressSanitizer: heap-use-after-free decode_prepared_descriptors"),
                    ("undefined", "runtime error: signed integer overflow"))
        self.subtest_cases = ("memory-zero-exit", "undefined-zero-exit")
        for case_id, (mode, output) in zip(self.subtest_cases, controls):
            with self.subTest(case=case_id), self.assertRaises(ValueError):
                verify_failure(mode, 0, output)


SanitizerControlTests.test_memory_requires_the_production_decoder_and_address_report.required_subtests = True
SanitizerControlTests.test_memory_requires_the_production_decoder_and_address_report.subtest_cases = (
    "empty", "signal-only", "address-report-only", "decoder-only", "wrong-address-report"
)
SanitizerControlTests.test_success_never_qualifies_a_control.required_subtests = True
SanitizerControlTests.test_success_never_qualifies_a_control.subtest_cases = (
    "memory-zero-exit", "undefined-zero-exit"
)


if __name__ == "__main__":
    unittest.main()
