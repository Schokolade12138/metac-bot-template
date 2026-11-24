import unittest

from bot.free_research import (
    dedupe_headlines,
    fallback_queries,
    parse_google_news_rss,
    parse_manifold,
    parse_polymarket,
    parse_queries,
    render,
)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>q</title>
<item><title>ECB holds rates steady - Reuters</title><link>https://example.com/a</link>
<pubDate>Fri, 20 Nov 2026 10:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>ECB holds rates steady - Reuters</title><link>https://example.com/b</link>
<pubDate>Thu, 19 Nov 2026 10:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>Lagarde signals patience</title><link>https://example.com/c</link>
<pubDate>Sat, 21 Nov 2026 08:30:00 GMT</pubDate><source url="https://ft.com">FT</source></item>
<item><title>No date here</title><pubDate>not a date</pubDate></item>
</channel></rss>"""


class GoogleNewsTest(unittest.TestCase):
    def test_parses_items_and_strips_source_suffix(self):
        headlines = parse_google_news_rss(RSS)
        self.assertEqual(len(headlines), 3)
        self.assertEqual(headlines[0].title, "ECB holds rates steady")
        self.assertEqual(headlines[0].source, "Reuters")

    def test_dedupes_and_sorts_newest_first(self):
        unique = dedupe_headlines(parse_google_news_rss(RSS))
        self.assertEqual([h.title for h in unique], ["Lagarde signals patience", "ECB holds rates steady"])


class MarketsTest(unittest.TestCase):
    def test_manifold_keeps_only_open_binary_markets_with_traders(self):
        markets = parse_manifold(
            [
                {"question": "A?", "probability": 0.25, "uniqueBettorCount": 30, "closeTime": 1798675140000, "url": "u1"},
                {"question": "Thin?", "probability": 0.5, "uniqueBettorCount": 2, "url": "u2"},
                {"question": "Multi?", "outcomeType": "MULTIPLE_CHOICE", "uniqueBettorCount": 50, "url": "u3"},
                {"question": "Done?", "probability": 1, "isResolved": True, "uniqueBettorCount": 50, "url": "u4"},
            ]
        )
        self.assertEqual([m.question for m in markets], ["A?"])
        self.assertEqual(markets[0].odds, "Yes 25%")
        self.assertIn("closes 2026-12-30", markets[0].detail)

    def test_polymarket_skips_closed_and_small(self):
        payload = {
            "events": [
                {
                    "title": "Fed rate cut by...?", "slug": "fed", "volume": 3_900_000, "closed": False,
                    "markets": [
                        {"question": "Cut by January?", "outcomes": '["Yes", "No"]', "outcomePrices": '["0", "1"]', "closed": True},
                        {"question": "Cut by December?", "outcomes": '["Yes", "No"]', "outcomePrices": '["0.34", "0.66"]', "closed": False, "endDate": "2026-12-10T00:00:00Z"},
                        {"question": "Broken", "outcomes": "oops", "outcomePrices": "[]", "closed": False},
                    ],
                },
                {"title": "Tiny", "slug": "tiny", "volume": 10, "closed": False, "markets": [{"question": "x", "outcomes": '["Yes"]', "outcomePrices": '["1"]'}]},
            ]
        }
        markets = parse_polymarket(payload)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].question, "Cut by December?")
        self.assertEqual(markets[0].odds, "Yes 34%, No 66%")
        self.assertEqual(markets[0].url, "https://polymarket.com/event/fed")


class QueriesTest(unittest.TestCase):
    QUESTION = "Will the ECB cut its deposit rate before March 2027?"

    def test_reads_json_wrapped_in_prose(self):
        raw = 'Sure!\n```json\n{"news": ["ECB deposit rate decision", "Lagarde rate cut"], "markets": ["ECB rate cut"]}\n```'
        queries = parse_queries(raw, self.QUESTION)
        self.assertEqual(queries["news"], ["ECB deposit rate decision", "Lagarde rate cut"])
        self.assertEqual(queries["markets"], ["ECB rate cut"])

    def test_falls_back_to_keywords_on_garbage(self):
        self.assertEqual(parse_queries("no json", self.QUESTION), fallback_queries(self.QUESTION))
        self.assertEqual(parse_queries('{"news": "oops"}', self.QUESTION), fallback_queries(self.QUESTION))

    def test_fallback_drops_stopwords(self):
        queries = fallback_queries(self.QUESTION)
        self.assertEqual(queries["markets"], ["ECB cut deposit rate"])


class RenderTest(unittest.TestCase):
    def test_empty_research_still_names_failures(self):
        text = render([], [], ["Polymarket"])
        self.assertIn("No headlines found.", text)
        self.assertIn("No related markets found.", text)
        self.assertIn("could not be reached: Polymarket", text)


if __name__ == "__main__":
    unittest.main()
