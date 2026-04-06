import unittest

from scripts.normalize_and_filter_pairs import filter_and_normalize
from src.utils.schemas import TranslationRow


def make_row(source_ref: str, source_en: str, target_ja: str) -> TranslationRow:
    return TranslationRow(
        id=f"{source_ref}-row",
        source_en=source_en,
        target_ja=target_ja,
        source_ref=source_ref,
        quality_score=0.9,
        license="cc-by-4.0",
    )


class NormalizeAndFilterTests(unittest.TestCase):
    def test_filters_navigation_boilerplate_from_jparacrawl(self) -> None:
        row = make_row(
            "jparacrawl",
            (
                "1001166 1MEDIAWORLD Log In / CONTACT Company Products "
                "Video Newsletters Live Meetings Video Chat Sign-up Forms Pricing"
            ),
            "ログイン 会社 製品 ニュースレター ミーティング 価格",
        )

        kept, stats = filter_and_normalize(
            [row],
            min_en_chars=10,
            min_ja_chars=4,
            max_len_ratio=5.0,
            min_quality=0.5,
            max_en_chars=400,
            max_ja_chars=400,
        )

        self.assertEqual([], kept)
        self.assertEqual(1, stats["boilerplate_noise"])

    def test_filters_overlong_crawl_rows(self) -> None:
        row = make_row(
            "jparacrawl",
            "word " * 120,
            "あ" * 140,
        )

        kept, stats = filter_and_normalize(
            [row],
            min_en_chars=10,
            min_ja_chars=4,
            max_len_ratio=5.0,
            min_quality=0.5,
            max_en_chars=400,
            max_ja_chars=400,
        )

        self.assertEqual([], kept)
        self.assertEqual(1, stats["too_long"])

    def test_keeps_reasonable_sentence(self) -> None:
        row = make_row(
            "hf_tatoeba",
            "Please save the settings before you close the window.",
            "ウィンドウを閉じる前に設定を保存してください。",
        )

        kept, stats = filter_and_normalize(
            [row],
            min_en_chars=10,
            min_ja_chars=4,
            max_len_ratio=5.0,
            min_quality=0.5,
            max_en_chars=400,
            max_ja_chars=400,
        )

        self.assertEqual(1, len(kept))
        self.assertEqual(1, stats["output"])


if __name__ == "__main__":
    unittest.main()
