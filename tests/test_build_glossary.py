import unittest

from scripts.build_glossary import build_glossary_rows


class BuildGlossaryTests(unittest.TestCase):
    def test_curated_only_skips_data_derived_terms(self) -> None:
        rows = [
            {"source_en": "Foobarterm setup guide", "target_ja": "フーバー項目"},
            {"source_en": "Foobarterm setup guide", "target_ja": "フーバー項目"},
            {"source_en": "Foobarterm setup guide", "target_ja": "フーバー項目"},
        ]

        derived_terms = {term[0] for term in build_glossary_rows(rows, min_freq=2, max_terms=50)}
        curated_only_terms = {
            term[0] for term in build_glossary_rows(rows, min_freq=2, max_terms=50, curated_only=True)
        }

        self.assertIn("foobarterm", derived_terms)
        self.assertNotIn("foobarterm", curated_only_terms)


if __name__ == "__main__":
    unittest.main()
