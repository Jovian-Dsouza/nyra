import unittest

from orchestrator.hermes_client import SentenceAggregator


class SentenceAggregatorTests(unittest.TestCase):
    def test_emits_on_punctuation(self) -> None:
        agg = SentenceAggregator()
        chunks = agg.push("It is 22 degrees. Expect rain")
        self.assertEqual(chunks, ["It is 22 degrees."])
        self.assertEqual(agg.flush_tail(), "Expect rain")

    def test_handles_multiple_sentences(self) -> None:
        agg = SentenceAggregator()
        chunks = agg.push("First. Second! Third?")
        self.assertEqual(chunks, ["First.", "Second!", "Third?"])
        self.assertIsNone(agg.flush_tail())


if __name__ == "__main__":
    unittest.main()

