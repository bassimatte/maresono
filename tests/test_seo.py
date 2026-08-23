import json
import re
import unittest
from pathlib import Path


CANONICAL_URL = "https://bassimatte.github.io/maresono/"
EXPECTED_TITLE = "Maresono — Generative Ocean Sound Synthesizer"


class SeoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_html = Path("engine/static/index.html").read_text(encoding="utf-8")
        cls.docs_html = Path("docs/index.html").read_text(encoding="utf-8")

    def test_static_and_deployed_interfaces_match(self):
        self.assertEqual(self.static_html, self.docs_html)

    def test_search_metadata_is_complete(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn(f"<title>{EXPECTED_TITLE}</title>", html)
            self.assertRegex(
                html,
                r'<meta name="description" content="[^\"]{70,160}" />',
            )
            self.assertIn('<meta name="robots" content="index, follow" />', html)
            self.assertIn(f'<link rel="canonical" href="{CANONICAL_URL}" />', html)

    def test_social_metadata_matches_the_canonical_page(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn('<meta property="og:type" content="website" />', html)
            self.assertIn(f'<meta property="og:url" content="{CANONICAL_URL}" />', html)
            self.assertIn(f'<meta property="og:title" content="{EXPECTED_TITLE}" />', html)
            self.assertIn('<meta name="twitter:card" content="summary" />', html)
            self.assertIn(
                f'<meta name="twitter:title" content="{EXPECTED_TITLE}" />', html
            )

    def test_web_application_structured_data_is_valid_and_honest(self):
        match = re.search(
            r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
            self.static_html,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        data = json.loads(match.group(1))

        self.assertEqual(data["@context"], "https://schema.org")
        self.assertEqual(data["@type"], "WebApplication")
        self.assertEqual(data["name"], "Maresono")
        self.assertEqual(data["url"], CANONICAL_URL)
        self.assertTrue(data["isAccessibleForFree"])
        self.assertEqual(data["offers"]["price"], "0")
        self.assertNotIn("aggregateRating", data)
        self.assertNotIn("review", data)

    def test_product_description_is_visible(self):
        for html in (self.static_html, self.docs_html):
            self.assertRegex(
                html,
                r'<p class="product-intro">[^<]*generative ocean sound synthesizer[^<]*</p>',
            )

    def test_favicon_is_deployed_with_both_frontends(self):
        static_favicon = Path("engine/static/favicon.svg")
        docs_favicon = Path("docs/favicon.svg")

        self.assertEqual(
            static_favicon.read_text(encoding="utf-8"),
            docs_favicon.read_text(encoding="utf-8"),
        )
        self.assertIn('href="./favicon.svg"', self.static_html)
        self.assertIn('@app.get("/favicon.svg", include_in_schema=False)', Path(
            "engine/web_server.py"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
