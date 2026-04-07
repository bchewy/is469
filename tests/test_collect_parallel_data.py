import unittest

from scripts.collect_parallel_data import (
    _jparacrawl_item_accepted,
    _parse_hf_tatoeba_tsv,
)


class JParaCrawlAcceptanceTests(unittest.TestCase):
    def test_accepts_single_model_vote_by_default(self) -> None:
        item = {"model1_accepted": 1, "model2_accepted": 0}

        self.assertTrue(_jparacrawl_item_accepted(item, require_both=False))

    def test_requires_both_votes_when_requested(self) -> None:
        item = {"model1_accepted": 1, "model2_accepted": 0}

        self.assertFalse(_jparacrawl_item_accepted(item, require_both=True))

    def test_accepts_two_votes_in_strict_mode(self) -> None:
        item = {"model1_accepted": 1, "model2_accepted": 1}

        self.assertTrue(_jparacrawl_item_accepted(item, require_both=True))

    def test_treats_string_zero_flags_as_false(self) -> None:
        item = {"model1_accepted": "0", "model2_accepted": "0"}

        self.assertFalse(_jparacrawl_item_accepted(item, require_both=False))

    def test_parses_hf_tatoeba_tsv_pairs(self) -> None:
        rows = _parse_hf_tatoeba_tsv(
            "eng\tjpn_Bopo\tHello there\tこんにちは\neng\tjpn_Bopo\tGoodbye\tさようなら\n",
            max_rows=1,
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("Hello there", rows[0].source_en)
        self.assertEqual("こんにちは", rows[0].target_ja)
        self.assertEqual("hf_tatoeba", rows[0].source_ref)


if __name__ == "__main__":
    unittest.main()
