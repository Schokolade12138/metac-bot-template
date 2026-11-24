# How this bot works

This fork runs a forecasting bot (`sakuraqkl`) in the Metaculus FutureEval bot
tournaments. It is fully automated: no human reviews or edits a forecast.

## Pipeline

For every open tournament question the bot has not forecast yet:

1. **Research** (`bot/forecaster.py`, `bot/free_research.py`), all sources in parallel:
   - recent headlines from Google News RSS, using search queries written by a small model;
   - prices of possibly related prediction markets on Polymarket and Manifold;
   - a web-search summary from a search-enabled model (paid profile only);
   - AskNews summaries, when AskNews credentials are configured.
   A source that fails is skipped; research never blocks a forecast.
2. **Forecast**: the question and research go through the template's prompts
   (`main.py`). Several forecasts are made per question, each by the next model in
   the configured list, so models from different labs vote. A failed forecast is
   retried once on the next model.
3. **Guards** (`bot/guards.py`): binary probabilities are kept within 2% to 98%,
   every multiple-choice option gets at least 1%.
4. **Aggregate and submit**: the framework takes the median and submits it, with
   the full reasoning of every forecast as a private comment.

## Configuration

Everything that decides what runs is in `bot/settings.py`: the model lists, the
number of forecasts per question, and the credit thresholds. `ACTIVE_PROFILE`
selects between:

- `free`: OpenRouter's zero-cost models. Used for the smoke test only. In
  tournament mode this profile lists the open questions and forecasts nothing.
- `paid`: donated tournament credits. Below $8 of remaining credits the bot
  switches to cheap models; below $0.50 it stops forecasting.

## Workflows

- `forecast.yaml`: the scheduled run (`run_bot.py --mode tournament`).
- `smoke_test.yaml`: manual; unit tests, then real forecasts on a few questions
  of the bot-testing-area tournament.
- `check_setup.yaml`: manual; checks the secrets without printing them.
- The template's own workflows are kept as manual fallbacks without schedules.

Runs exit successfully even when a question fails; failures are listed in the
run summary and the question is retried by the next scheduled run.

## Tests

    uv run python -m unittest discover -s tests -t .
