"""
Research that costs nothing: recent headlines from Google News RSS and prices
from public prediction markets (Polymarket, Manifold). No API keys needed.

Every source is optional. A source that fails or times out is skipped and named
in the output, so a forecast is never blocked by a research outage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from forecasting_tools import GeneralLlm, MetaculusQuestion, clean_indents

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; forecasting-research-bot)"}
MAX_HEADLINES = 30
MAX_MARKETS_PER_SOURCE = 6
_STOPWORDS = set(
    "a an and are as at be before between by for from has have how in is it its many "
    "much of on or than that the their there this to was what when which who will with "
    "would after during than more less least most following any".split()
)


@dataclass(frozen=True)
class Headline:
    published: datetime
    source: str
    title: str


@dataclass(frozen=True)
class Market:
    venue: str
    question: str
    odds: str
    detail: str
    url: str


# ------------------------------------------------------------------ queries
def fallback_queries(question_text: str) -> dict[str, list[str]]:
    """Keyword queries built without an LLM, used when query writing fails."""
    words = re.findall(r"[A-Za-z0-9$%.\-']+", question_text)
    keywords = [w for w in words if w.lower() not in _STOPWORDS]
    news = " ".join(keywords[:9]) or question_text[:80]
    markets = " ".join(keywords[:4]) or question_text[:40]
    return {"news": [news], "markets": [markets]}


def parse_queries(raw: str, question_text: str) -> dict[str, list[str]]:
    fallback = fallback_queries(question_text)
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        return fallback
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback

    def clean(key: str, limit: int) -> list[str]:
        values = data.get(key) if isinstance(data, dict) else None
        if not isinstance(values, list):
            return []
        cleaned = [re.sub(r"\s+", " ", str(v)).strip(" \"'") for v in values]
        return [v for v in cleaned if 2 < len(v) <= 120][:limit]

    return {
        "news": clean("news", 3) or fallback["news"],
        "markets": clean("markets", 2) or fallback["markets"],
    }


async def write_queries(
    question: MetaculusQuestion, llm: GeneralLlm | None
) -> dict[str, list[str]]:
    if llm is None:
        return fallback_queries(question.question_text)
    prompt = clean_indents(
        f"""
        A forecaster needs search queries for this question.

        Question: {question.question_text}
        Resolution criteria: {(question.resolution_criteria or "")[:1500]}

        Reply with JSON only, in exactly this shape:
        {{"news": ["...", "...", "..."], "markets": ["...", "..."]}}

        - "news": three different Google News queries of 3 to 7 words each, plain
          keywords without quotes or operators. Aim at the latest developments
          that decide the outcome, and name the key people, organisations or
          indicators.
        - "markets": two keyword phrases of 2 to 4 words each for finding
          prediction markets on the same or a closely related event.
        """
    )
    try:
        return parse_queries(await llm.invoke(prompt), question.question_text)
    except Exception as error:
        logger.warning(f"Query writing failed, using keywords: {type(error).__name__}")
        return fallback_queries(question.question_text)


# ------------------------------------------------------------------ sources
def parse_google_news_rss(xml_text: str) -> list[Headline]:
    headlines: list[Headline] = []
    for item in ElementTree.fromstring(xml_text).iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        if title:
            headlines.append(Headline(published, source or "unknown", title))
    return headlines


async def google_news(client: httpx.AsyncClient, query: str) -> list[Headline]:
    response = await client.get(
        "https://news.google.com/rss/search",
        params={"q": f"{query} when:30d", "hl": "en-US", "gl": "US", "ceid": "US:en"},
    )
    response.raise_for_status()
    return parse_google_news_rss(response.text)


def parse_manifold(markets: list[dict]) -> list[Market]:
    parsed: list[Market] = []
    for market in markets:
        probability = market.get("probability")
        traders = market.get("uniqueBettorCount") or 0
        if market.get("isResolved") or probability is None or traders < 8:
            continue
        parsed.append(
            Market(
                venue="Manifold (play money)",
                question=str(market.get("question", "")).strip(),
                odds=f"Yes {probability * 100:.0f}%",
                detail=f"{traders} traders, closes {_ms_to_date(market.get('closeTime'))}",
                url=str(market.get("url", "")),
            )
        )
    return parsed


async def manifold(client: httpx.AsyncClient, term: str) -> list[Market]:
    response = await client.get(
        "https://api.manifold.markets/v0/search-markets",
        params={"term": term, "limit": 8, "filter": "open"},
    )
    response.raise_for_status()
    return parse_manifold(response.json())


def parse_polymarket(payload: dict) -> list[Market]:
    parsed: list[Market] = []
    for event in payload.get("events") or []:
        if event.get("closed") or (event.get("volume") or 0) < 5_000:
            continue
        open_markets = [m for m in event.get("markets") or [] if not m.get("closed")]
        for market in open_markets[:4]:
            try:
                outcomes = json.loads(market.get("outcomes") or "[]")
                prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
            except (TypeError, ValueError):
                continue
            if not outcomes or len(outcomes) != len(prices):
                continue
            odds = ", ".join(f"{o} {p * 100:.0f}%" for o, p in zip(outcomes, prices))
            parsed.append(
                Market(
                    venue="Polymarket (real money)",
                    question=str(market.get("question") or event.get("title", "")).strip(),
                    odds=odds,
                    detail=(
                        f"event volume ${float(event.get('volume') or 0):,.0f}, "
                        f"closes {str(market.get('endDate') or event.get('endDate') or '?')[:10]}"
                    ),
                    url=f"https://polymarket.com/event/{event.get('slug', '')}",
                )
            )
    return parsed


async def polymarket(client: httpx.AsyncClient, term: str) -> list[Market]:
    response = await client.get(
        "https://gamma-api.polymarket.com/public-search",
        params={"q": term, "limit_per_type": 4, "events_status": "active"},
    )
    response.raise_for_status()
    return parse_polymarket(response.json())


def _ms_to_date(milliseconds: object) -> str:
    try:
        return datetime.fromtimestamp(float(milliseconds) / 1000, timezone.utc).strftime("%Y-%m-%d")  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


# ------------------------------------------------------------------ assembly
def dedupe_headlines(headlines: list[Headline], limit: int = MAX_HEADLINES) -> list[Headline]:
    seen: set[str] = set()
    unique: list[Headline] = []
    for headline in sorted(headlines, key=lambda h: h.published, reverse=True):
        key = re.sub(r"[^a-z0-9]+", " ", headline.title.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(headline)
    return unique[:limit]


def dedupe_markets(markets: list[Market], limit: int = MAX_MARKETS_PER_SOURCE) -> list[Market]:
    seen: set[str] = set()
    unique: list[Market] = []
    for market in markets:
        if market.url + market.question not in seen:
            seen.add(market.url + market.question)
            unique.append(market)
    return unique[:limit]


def render(
    headlines: list[Headline],
    markets: list[Market],
    failed_sources: list[str],
) -> str:
    lines: list[str] = ["## Recent headlines (Google News, last 30 days, newest first)"]
    if headlines:
        lines += [
            f"- {h.published.strftime('%Y-%m-%d')} | {h.source} | {h.title}"
            for h in headlines
        ]
    else:
        lines.append("No headlines found.")
    lines += [
        "",
        "## Prediction markets found by keyword search",
        "These may differ from the question in wording, deadline or resolution "
        "source. Use a price only where the market clearly matches the question.",
    ]
    if markets:
        lines += [
            f"- {m.venue} | {m.question} | {m.odds} | {m.detail} | {m.url}"
            for m in markets
        ]
    else:
        lines.append("No related markets found.")
    if failed_sources:
        lines += ["", f"Sources that could not be reached: {', '.join(failed_sources)}."]
    return "\n".join(lines)


async def gather_free_research(
    question: MetaculusQuestion, query_llm: GeneralLlm | None
) -> str:
    queries = await write_queries(question, query_llm)
    async with httpx.AsyncClient(
        timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True
    ) as client:
        jobs = (
            [("Google News", google_news(client, q)) for q in queries["news"]]
            + [("Polymarket", polymarket(client, q)) for q in queries["markets"]]
            + [("Manifold", manifold(client, q)) for q in queries["markets"]]
        )
        results = await asyncio.gather(*(job for _, job in jobs), return_exceptions=True)

    headlines: list[Headline] = []
    by_venue: dict[str, list[Market]] = {"Polymarket": [], "Manifold": []}
    failed: list[str] = []
    for (name, _), result in zip(jobs, results):
        if isinstance(result, BaseException):
            logger.warning(f"{name} lookup failed: {type(result).__name__}: {result}")
            if name not in failed:
                failed.append(name)
        elif name == "Google News":
            headlines += result
        else:
            by_venue[name] += result
    markets = dedupe_markets(by_venue["Polymarket"]) + dedupe_markets(by_venue["Manifold"])
    return render(dedupe_headlines(headlines), markets, failed)
