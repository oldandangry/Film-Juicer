"""Independent finite tooling controls for the synthetic policy recipe."""

import ast
import hashlib
import json
from pathlib import Path
import unittest


class MiniatureToolingTests(unittest.TestCase):
    def test_known_hash_and_json_integrity(self):
        self.assertEqual(hashlib.sha256(b"abc").hexdigest(),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(json.loads('{"scope":"synthetic","members":[1,2]}'),
                         {"scope": "synthetic", "members": [1, 2]})

    def test_fixed_checker_has_independent_closed_policy(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "scripts/check-quality.py").read_bytes())
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in {"SOURCE_FILES", "POLICY"}}
        self.assertEqual(len(constants["SOURCE_FILES"]), 12)
        self.assertEqual(constants["POLICY"]["checks"], ["diff", "syntax", "ruff", "format", "tidy"])
        self.assertIn("--isolated", constants["POLICY"]["ruff"])
        self.assertIn("--no-cache", constants["POLICY"]["ruff"])
        self.assertIn("InheritParentConfig: false", constants["POLICY"]["tidy"])
        for name in constants["SOURCE_FILES"]:
            if name.endswith(".py"):
                ast.parse((root / name).read_bytes(), filename=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
