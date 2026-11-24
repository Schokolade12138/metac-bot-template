"""
What the bot runs with: models, number of forecasts per question and budget limits.

Two profiles exist, both through the OPENROUTER_API_KEY secret:
- "paid": donated tournament credits. Several mid-priced reasoning models from
  different labs each forecast the question and the median is submitted.
- "free": OpenRouter's zero-cost ":free" models. Small daily quotas and weaker
  models, used to prove the pipeline end to end without spending anything.

ACTIVE_PROFILE decides which one runs. Change the constants in this file (or set
the matching environment variable) to change what runs. Nothing else in the bot
hard-codes a model name.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx
from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- paid profile
PAID_FORECAST_MODELS = [
    "openrouter/openai/gpt-6.2-sol",
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/google/gemini-4-carbon",
]
# Used instead of the list above once remaining credits drop under LOW_CREDIT_USD.
PAID_CHEAP_FORECAST_MODELS = [
    "openrouter/openai/gpt-6-luna",
    "openrouter/anthropic/claude-haiku-5.5",
]
PAID_UTILITY_MODEL = "openrouter/openai/gpt-6-luna"  # parsing, search queries
PAID_SEARCH_MODEL = "openrouter/openai/gpt-6.1-terra:online"  # web-search research
PAID_PREDICTIONS_PER_QUESTION = 3
LOW_CREDIT_USD = 8.0  # below this: cheap models, no paid web search
MIN_CREDIT_USD = 0.5  # below this: do not forecast at all

# ---------------------------------------------------------------- free profile
FREE_FORECAST_MODELS = [
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/baidu/ernie-5.5-flash-preview:free",
    "openrouter/inclusionai/ling-3.2-flash:free",
]
FREE_UTILITY_MODEL = "openrouter/google/gemma-4-31b-it:free"
FREE_PREDICTIONS_PER_QUESTION = 3
FREE_RESEARCH_CHAR_BUDGET = 20_000
PAID_RESEARCH_CHAR_BUDGET = 40_000

# Which profile runs when BOT_PROFILE is not set. Stays "free" until a key with
# tournament credits is in place, so a key without credits never tries paid models.
ACTIVE_PROFILE = "free"

_PLACEHOLDERS = {"", "1234567890", "REPLACE_ME", "your-token-here", "your-api-key-here"}


def _env(name: str) -> str | None:
    value = (os.getenv(name) or "").strip()
    return None if value in _PLACEHOLDERS else value


def _env_list(name: str, default: list[str]) -> list[str]:
    value = _env(name)
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    profile: str  # "paid" or "free"
    forecast_llms: list[GeneralLlm]
    utility_llm: GeneralLlm
    search_llm: GeneralLlm | None
    predictions_per_question: int
    research_char_budget: int
    question_concurrency: int
    credits_remaining_usd: float | None = None
    notes: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def forecast_model_names(self) -> list[str]:
        return [llm.model for llm in self.forecast_llms]


def _openrouter_llm(model: str, *, reasoning: bool, timeout: int) -> GeneralLlm:
    kwargs: dict = {"timeout": timeout, "allowed_tries": 2}
    if reasoning:
        kwargs["reasoning_effort"] = _env("BOT_REASONING_EFFORT") or "high"
    else:
        kwargs["temperature"] = 0.2
    return GeneralLlm(model=model, **kwargs)


def fetch_openrouter_credits(api_key: str) -> float | None:
    """Remaining USD on the key, or None when unlimited or the lookup failed."""
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        response.raise_for_status()
        remaining = response.json().get("data", {}).get("limit_remaining")
        return None if remaining is None else float(remaining)
    except Exception as error:  # the credit check must never stop a forecast run
        logger.warning(f"Could not read OpenRouter credits: {type(error).__name__}")
        return None


def resolve_settings() -> Settings:
    profile = (_env("BOT_PROFILE") or ACTIVE_PROFILE).lower()
    api_key = _env("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "No LLM access: the OPENROUTER_API_KEY secret is missing. Add it under "
            "Settings > Secrets and variables > Actions."
        )
    if profile == "paid":
        return _paid_settings(api_key)
    if profile == "free":
        return _free_settings()
    raise SystemExit(f"Unknown BOT_PROFILE: {profile}")


def _paid_settings(api_key: str) -> Settings:
    credits = fetch_openrouter_credits(api_key)
    notes: list[str] = []
    skip_reason = None
    forecast_models = _env_list("BOT_FORECAST_MODELS", PAID_FORECAST_MODELS)
    search_model: str | None = _env("BOT_SEARCH_MODEL") or PAID_SEARCH_MODEL
    if credits is not None and credits < MIN_CREDIT_USD:
        skip_reason = f"Only ${credits:.2f} of credits left, not forecasting."
    elif credits is not None and credits < LOW_CREDIT_USD:
        forecast_models = list(PAID_CHEAP_FORECAST_MODELS)
        search_model = None
        notes.append(f"Low credits (${credits:.2f}): cheap models, no paid search.")
    return Settings(
        profile="paid",
        forecast_llms=[
            _openrouter_llm(model, reasoning=True, timeout=600)
            for model in forecast_models
        ],
        utility_llm=_openrouter_llm(
            _env("BOT_UTILITY_MODEL") or PAID_UTILITY_MODEL, reasoning=False, timeout=90
        ),
        search_llm=(
            _openrouter_llm(search_model, reasoning=False, timeout=180)
            if search_model
            else None
        ),
        predictions_per_question=int(
            _env("BOT_PREDICTIONS") or PAID_PREDICTIONS_PER_QUESTION
        ),
        research_char_budget=PAID_RESEARCH_CHAR_BUDGET,
        question_concurrency=3,
        credits_remaining_usd=credits,
        notes=notes,
        skip_reason=skip_reason,
    )


def _free_settings() -> Settings:
    forecast_models = _env_list("BOT_FORECAST_MODELS", FREE_FORECAST_MODELS)
    return Settings(
        profile="free",
        forecast_llms=[
            _openrouter_llm(model, reasoning=False, timeout=240)
            for model in forecast_models
        ],
        utility_llm=_openrouter_llm(
            _env("BOT_UTILITY_MODEL") or FREE_UTILITY_MODEL, reasoning=False, timeout=120
        ),
        search_llm=None,
        predictions_per_question=int(
            _env("BOT_PREDICTIONS") or FREE_PREDICTIONS_PER_QUESTION
        ),
        research_char_budget=FREE_RESEARCH_CHAR_BUDGET,
        question_concurrency=1,
    )
