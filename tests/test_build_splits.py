import unittest
from collections import Counter

from scripts.build_splits import build_splits
from src.utils.schemas import TranslationRow


def make_rows(source_ref: str, count: int) -> list[TranslationRow]:
    return [
        TranslationRow(
            id=f"{source_ref}-{i}",
            source_en=f"Example sentence {i} from {source_ref}",
            target_ja=f"{source_ref} の例文 {i}",
            source_ref=source_ref,
            quality_score=0.9,
            license="cc-by-4.0",
        )
        for i in range(count)
    ]


class BuildSplitsTests(unittest.TestCase):
    def test_applies_source_train_quotas_and_allows_empty_test(self) -> None:
        rows = (
            make_rows("jparacrawl", 20)
            + make_rows("hf_tatoeba", 10)
            + make_rows("opus100_filtered", 6)
        )

        splits = build_splits(
            rows,
            train_ratio=0.9,
            dev_ratio=0.1,
            test_ratio=0.0,
            seed=42,
            source_train_quotas={
                "jparacrawl": 9,
                "hf_tatoeba": 4,
                "opus100_filtered": 2,
            },
            allow_empty_test=True,
        )

        train_counts = Counter(row.source_ref for row in splits["train"])
        dev_counts = Counter(row.source_ref for row in splits["dev"])

        self.assertEqual(
            {"jparacrawl": 9, "hf_tatoeba": 4, "opus100_filtered": 2},
            dict(train_counts),
        )
        self.assertEqual(0, len(splits["test"]))
        self.assertGreaterEqual(dev_counts["jparacrawl"], 1)
        self.assertGreaterEqual(dev_counts["hf_tatoeba"], 1)
        self.assertGreaterEqual(dev_counts["opus100_filtered"], 1)

    def test_normalizes_non_unit_ratios_in_quota_mode(self) -> None:
        rows = make_rows("jparacrawl", 12)

        splits = build_splits(
            rows,
            train_ratio=80,
            dev_ratio=10,
            test_ratio=10,
            seed=42,
            source_train_quotas={"jparacrawl": 8},
            allow_empty_test=False,
        )

        self.assertEqual(8, len(splits["train"]))
        self.assertEqual(1, len(splits["dev"]))
        self.assertEqual(1, len(splits["test"]))

    def test_rejects_unlisted_sources_in_quota_mode(self) -> None:
        rows = make_rows("jparacrawl", 10) + make_rows("unexpected_source", 3)

        with self.assertRaises(ValueError):
            build_splits(
                rows,
                train_ratio=0.9,
                dev_ratio=0.1,
                test_ratio=0.0,
                seed=42,
                source_train_quotas={"jparacrawl": 9},
                allow_empty_test=True,
            )


if __name__ == "__main__":
    unittest.main()
