"""
Entry point for the tournament bot.

    uv run python run_bot.py --mode tournament       # seasonal tournament + MiniBench
    uv run python run_bot.py --mode test_questions   # bot-testing-area smoke test

The free profile never forecasts in the prize tournaments: in tournament mode it
only lists the open questions, so the schedule can be watched without publishing
forecasts from weak models. Tournament forecasting starts with the paid profile.

Runs must not fail the workflow (GitHub emails the repository owner on every
failure): a question that errors is picked up again by the next scheduled run,
and problems are reported in the run summary instead. Pass --strict when running
by hand to get a non-zero exit code if nothing was forecast.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import traceback
from typing import Sequence

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import ForecastReport, MetaculusClient, MetaculusQuestion

from bot.forecaster import TournamentBot
from bot.settings import Settings, resolve_settings

logger = logging.getLogger("run_bot")

TEST_TOURNAMENT = "bot-testing-area"


def pick_test_questions(
    questions: Sequence[MetaculusQuestion], limit: int
) -> list[MetaculusQuestion]:
    """A small sample that covers as many question types as possible."""
    by_type: dict[type, list[MetaculusQuestion]] = {}
    for question in questions:
        by_type.setdefault(type(question), []).append(question)
    picked: list[MetaculusQuestion] = []
    while len(picked) < limit and any(by_type.values()):
        for group in by_type.values():
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    return picked


def load_questions(
    mode: str, client: MetaculusClient, limit: int | None
) -> list[MetaculusQuestion]:
    if mode == "test_questions":
        questions = client.get_all_open_questions_from_tournament(TEST_TOURNAMENT)
        return pick_test_questions(questions, limit or 4)

    questions: list[MetaculusQuestion] = []
    for tournament in (client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID):
        try:
            questions += client.get_all_open_questions_from_tournament(tournament)
        except Exception as error:
            logger.warning(f"Could not load {tournament}: {type(error).__name__}: {error}")
    new_questions = [q for q in questions if not q.already_forecasted]
    logger.info(f"{len(questions)} open questions, {len(new_questions)} not yet forecast")
    return new_questions[:limit] if limit else new_questions


def describe_question(question: MetaculusQuestion) -> str:
    def short(moment: object) -> str:
        return moment.strftime("%m-%d %H:%M") if hasattr(moment, "strftime") else "?"  # type: ignore[union-attr]

    return (
        f"{type(question).__name__} opened {short(question.open_time)} "
        f"closes {short(question.close_time)} UTC: {question.page_url}"
    )


async def forecast_all(
    bot: TournamentBot, questions: Sequence[MetaculusQuestion], concurrency: int
) -> list[ForecastReport | BaseException]:
    semaphore = asyncio.Semaphore(concurrency)

    async def forecast_one(question: MetaculusQuestion) -> ForecastReport | BaseException:
        async with semaphore:
            reports = await bot.forecast_questions([question], return_exceptions=True)
            return reports[0]

    return list(await asyncio.gather(*(forecast_one(q) for q in questions)))


def write_summary(
    mode: str,
    settings: Settings | None,
    questions: Sequence[MetaculusQuestion],
    reports: Sequence[ForecastReport | BaseException],
    notes: Sequence[str] = (),
) -> str:
    done = [r for r in reports if isinstance(r, ForecastReport)]
    failed = [
        (question, report)
        for question, report in zip(questions, reports)
        if isinstance(report, BaseException)
    ]
    lines = [f"## Forecast run ({mode})"]
    if settings is not None:
        lines += [
            f"- Profile: {settings.profile}",
            f"- Forecast models: {', '.join(settings.forecast_model_names)}",
            f"- Forecasts per question: {settings.predictions_per_question}",
        ]
        if settings.credits_remaining_usd is not None:
            lines.append(f"- Credits remaining: ${settings.credits_remaining_usd:.2f}")
        lines += [f"- Note: {note}" for note in settings.notes]
        if settings.skip_reason:
            lines.append(f"- Skipped: {settings.skip_reason}")
    lines += [f"- Note: {note}" for note in notes]
    if reports:
        lines.append(f"- Questions forecast: {len(done)} of {len(questions)}")
    else:
        lines.append(f"- Questions waiting for a forecast: {len(questions)}")
        lines += [f"- WAITING: {describe_question(q)}" for q in questions]
    lines += [f"- OK: {report.question.page_url}" for report in done]
    lines += [
        f"- FAILED: {question.page_url}: {type(error).__name__}: {str(error)[:300]}"
        for question, error in failed
    ]
    summary = "\n".join(lines)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(summary + "\n")
    print(summary)
    return summary


def run(args: argparse.Namespace) -> int:
    if not (os.getenv("METACULUS_TOKEN") or "").strip():
        raise SystemExit("The METACULUS_TOKEN secret is missing.")
    notes: list[str] = []
    settings: Settings | None = None
    try:
        settings = resolve_settings()
    except SystemExit as error:
        notes.append(str(error))

    client = MetaculusClient()
    questions = load_questions(args.mode, client, args.limit)

    listing_only = (
        settings is None
        or settings.skip_reason is not None
        or (args.mode == "tournament" and settings.profile == "free")
    )
    if settings is not None and settings.profile == "free" and args.mode == "tournament":
        notes.append("Free profile: tournament questions are listed, not forecast.")
    if listing_only or not questions:
        write_summary(args.mode, settings, questions, [], notes)
        return 1 if args.strict else 0

    assert settings is not None
    bot = TournamentBot(
        settings,
        publish_reports_to_metaculus=not args.no_publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,  # already filtered above
        extra_metadata_in_explanation=True,
        metaculus_client=client,
    )
    reports = asyncio.run(forecast_all(bot, questions, settings.question_concurrency))
    bot.log_report_summary(reports, raise_errors=False)
    write_summary(args.mode, settings, questions, reports, notes)

    succeeded = sum(isinstance(report, ForecastReport) for report in reports)
    return 1 if args.strict and succeeded == 0 else 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the tournament forecasting bot")
    parser.add_argument(
        "--mode", choices=["tournament", "test_questions"], default="tournament"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max questions this run")
    parser.add_argument(
        "--strict", action="store_true", help="Exit non-zero if nothing was forecast"
    )
    parser.add_argument(
        "--no-publish", action="store_true", help="Forecast without submitting"
    )
    args = parser.parse_args()
    try:
        return run(args)
    except BaseException as error:  # includes SystemExit: see the module docstring
        if args.strict:
            raise
        traceback.print_exc()
        message = f"## Forecast run ({args.mode})\n- CRASHED: {type(error).__name__}: {str(error)[:500]}"
        summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as summary_file:
                summary_file.write(message + "\n")
        print(message)
        return 0


if __name__ == "__main__":
    sys.exit(main())
