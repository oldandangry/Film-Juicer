from __future__ import annotations

import unittest

from run_prepared_sanitizer_control import verify_failure


class SanitizerControlTests(unittest.TestCase):
    def test_memory_requires_the_production_decoder_and_address_report(self):
        verify_failure("memory", 1, "AddressSanitizer: heap-use-after-free\ndecode_prepared_descriptors")
        for output in ("", "segmentation fault", "AddressSanitizer: heap-use-after-free",
                       "decode_prepared_descriptors", "AddressSanitizer: heap-buffer-overflow\ndecode_prepared_descriptors"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                verify_failure("memory", 1, output)

    def test_undefined_requires_signed_overflow_report(self):
        verify_failure("undefined", 1, "runtime error: signed integer overflow")
        with self.assertRaises(ValueError):
            verify_failure("undefined", 1, "UndefinedBehaviorSanitizer: other error")

    def test_success_never_qualifies_a_control(self):
        for mode, output in (("memory", "AddressSanitizer: heap-use-after-free decode_prepared_descriptors"),
                             ("undefined", "runtime error: signed integer overflow")):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verify_failure(mode, 0, output)


if __name__ == "__main__":
    unittest.main()
