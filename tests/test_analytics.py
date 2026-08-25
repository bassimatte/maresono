import re
import unittest
from pathlib import Path


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_html = Path("engine/static/index.html").read_text(encoding="utf-8")
        cls.docs_html = Path("docs/index.html").read_text(encoding="utf-8")

    def test_static_and_deployed_interfaces_match(self):
        self.assertEqual(self.static_html, self.docs_html)

    def test_umami_is_limited_to_canonical_maresono(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("const MARESONO_ANALYTICS_HOST = 'bassimatte.github.io';", html)
            self.assertIn("const MARESONO_ANALYTICS_PATH = '/maresono';", html)
            self.assertIn("const MARESONO_ANALYTICS_TAG = 'maresono';", html)
            self.assertIn("script.dataset.domains = MARESONO_ANALYTICS_HOST;", html)
            self.assertIn("script.dataset.tag = MARESONO_ANALYTICS_TAG;", html)

    def test_shared_website_id_and_event_prefix_are_configured(self):
        for html in (self.static_html, self.docs_html):
            match = re.search(r"const MARESONO_UMAMI_WEBSITE_ID = '([^']*)';", html)
            self.assertIsNotNone(match)
            self.assertRegex(
                match.group(1),
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            )
            self.assertIn(
                "window.umami.track(`${MARESONO_ANALYTICS_EVENT_PREFIX}${eventName}`, properties);",
                html,
            )

    def test_privacy_controls_are_enabled(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("script.dataset.excludeSearch = 'true';", html)
            self.assertIn("script.dataset.excludeHash = 'true';", html)
            self.assertIn("script.dataset.doNotTrack = 'true';", html)
            self.assertIn("if (!analyticsIsConfigured()) return false;", html)

    def test_product_events_are_wired(self):
        for event_name in (
            "play_requested",
            "audio_started",
            "audio_failed",
            "listening_reached",
            "export_started",
            "export_completed",
            "export_failed",
            "outbound_opened",
            "related_tool_opened",
            "portfolio_opened",
        ):
            self.assertIn(f"{event_name}:", self.static_html)
            self.assertRegex(self.static_html, rf"trackUsage\('{event_name}'")

    def test_dynamic_values_are_allowlisted(self):
        self.assertIn("if (allowed.includes(value)) props[key] = value;", self.static_html)
        calls = re.findall(
            r"trackUsage\('([^']+)'\s*,\s*\{(.*?)\}\);",
            self.static_html,
            re.DOTALL,
        )
        self.assertTrue(calls)
        property_text = "\n".join(properties for _, properties in calls)
        for forbidden_key in (
            "intensity:",
            "volume:",
            "filename:",
            "audio:",
            "error:",
            "message:",
            "sessionId:",
        ):
            self.assertNotIn(forbidden_key, property_text)

    def test_privacy_notice_is_visible(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("anonymous, aggregate usage statistics", html)
            self.assertIn("Maresono does not send intensity or volume values", html)
            self.assertIn("Analytics is disabled for local installations", html)

    def test_reciprocal_links_are_visible_and_tracked(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn('href="https://bassimatte.github.io/"', html)
            self.assertIn('href="https://github.com/bassimatte/maresono"', html)
            self.assertIn('data-analytics-destination="portfolio"', html)
            self.assertIn('data-analytics-destination="source"', html)
            self.assertIn("destination: ['portfolio', 'source', 'freesound']", html)

    def test_about_window_links_to_related_tools_without_self_linking(self):
        for html in (self.static_html, self.docs_html):
            related_line = re.search(
                r'<p class="related-tools">(.*?)</p>',
                html,
                re.DOTALL,
            )
            self.assertIsNotNone(related_line)
            links = related_line.group(1)

            for tool in ("mantice", "glorb", "campana"):
                self.assertIn(f'href="https://bassimatte.github.io/{tool}/"', links)
                self.assertIn(f'data-related-tool="{tool}"', links)
            self.assertIn('href="https://bassimatte.github.io/"', links)
            self.assertIn('data-portfolio-link="about"', links)
            self.assertNotIn("/maresono/", links)

    def test_about_links_use_dedicated_allowlisted_events(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("tool: ['mantice', 'glorb', 'campana']", html)
            self.assertIn("placement: ['about']", html)
            self.assertIn("trackUsage('related_tool_opened'", html)
            self.assertIn("trackUsage('portfolio_opened'", html)


if __name__ == "__main__":
    unittest.main()
