"""The complete native preparation replay has one finite integration owner."""

import unittest

import test_checkpoint


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([test_checkpoint.CheckpointTests("replay_preparation_renewal")])
