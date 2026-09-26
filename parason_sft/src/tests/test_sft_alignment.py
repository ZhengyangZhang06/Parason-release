import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PARASON_SFT = REPO_ROOT / "parason_sft"


class SftAlignmentTest(unittest.TestCase):
    def test_special_tokens_include_trial_and_subtask(self):
        source = (PARASON_SFT / "src" / "utils.py").read_text(encoding="utf-8")

        for token in ("<Trial>", "</Trial>", "<Subtask>", "</Subtask>"):
            self.assertIn(token, source)

    def test_prefix_tree_treats_trial_and_subtask_as_outline_aliases(self):
        source = (PARASON_SFT / "src" / "prefix_tree.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("TAG_TOKEN_ALIASES", source)
        self.assertRegex(
            source,
            re.compile(
                r"'Outline':\s*\[[^\]]*"
                r"\('<Outline>',\s*'</Outline>'\)[^\]]*"
                r"\('<Trial>',\s*'</Trial>'\)[^\]]*"
                r"\('<Subtask>',\s*'</Subtask>'\)",
                re.S,
            ),
        )

    def test_train_launcher_defaults_to_eight_epochs(self):
        source = (PARASON_SFT / "train_parallel.sh").read_text(encoding="utf-8")

        self.assertRegex(source, r"(?m)^epochs=8$")


if __name__ == "__main__":
    unittest.main()
