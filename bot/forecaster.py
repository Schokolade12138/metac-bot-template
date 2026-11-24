"""
The tournament bot: the template bot's prompts, with three changes around them.

1. Research comes from several independent sources instead of one.
2. Each forecast of a question is made by a different model where several are
   configured, and the framework submits the median.
3. Every forecast passes a guard before it counts: a failed attempt is retried on
   another model, and extreme probabilities are pulled back from 0 and 1.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from bot.free_research import gather_free_research
from bot.guards import clamp_probability, floor_option_probabilities
from bot.settings import Settings
from main import FallTemplateBot2026

logger = logging.getLogger(__name__)

FORECAST_ATTEMPTS = 2

# The model chosen for the forecast being made in the current asyncio task. The
# template asks get_llm("default") for its forecasting model; this lets each
# concurrent forecast get a different one.
_forecast_llm: ContextVar[GeneralLlm | None] = ContextVar("forecast_llm", default=None)


class TournamentBot(FallTemplateBot2026):
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        self.settings = settings
        self._forecast_cycle = itertools.cycle(settings.forecast_llms)
        if settings.profile == "free":
            # One parsing call per forecast instead of two, to fit free quotas.
            self._structure_output_validation_samples = 1
        super().__init__(
            research_reports_per_question=1,
            predictions_per_research_report=settings.predictions_per_question,
            use_research_summary_to_forecast=False,
            enable_summarize_research=False,
            parameters_to_exclude_from_config_dict=["settings"],
            llms={
                "default": settings.forecast_llms[0],
                "summarizer": settings.utility_llm,
                "parser": settings.utility_llm,
                "researcher": settings.search_llm or "no_research",
            },
            **kwargs,
        )

    def get_llm(self, purpose: str = "default", guarantee_type: Any = None) -> Any:
        chosen = _forecast_llm.get()
        if purpose == "default" and chosen is not None:
            return chosen.model if guarantee_type == "string_name" else chosen
        return super().get_llm(purpose, guarantee_type)

    # ------------------------------------------------------------ research
    async def run_research(self, question: MetaculusQuestion) -> str:
        sources: list[tuple[str, Awaitable[str]]] = []
        if self.settings.search_llm is not None:
            sources.append(("Web search summary", self._web_search(question)))
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            sources.append(("AskNews", self._asknews(question)))
        sources.append(
            ("Headlines and markets", gather_free_research(question, self.settings.utility_llm))
        )
        results = await asyncio.gather(
            *(job for _, job in sources), return_exceptions=True
        )
        sections: list[str] = []
        for (name, _), result in zip(sources, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Research source '{name}' failed for {question.page_url}: "
                    f"{type(result).__name__}: {result}"
                )
            elif result and result.strip():
                sections.append(f"# {name}\n{result.strip()}")
        research = "\n\n".join(sections)[: self.settings.research_char_budget]
        if not research:
            research = "No research could be retrieved for this question."
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research

    async def _web_search(self, question: MetaculusQuestion) -> str:
        llm = self.settings.search_llm
        assert llm is not None
        return await llm.invoke(self._get_research_prompt(question, llm))

    async def _asknews(self, question: MetaculusQuestion) -> str:
        return await AskNewsSearcher().call_preconfigured_version(
            "asknews/news-summaries", question.question_text
        )

    # ------------------------------------------------------------ forecasts
    async def _guarded(
        self,
        forecast: Callable[[Any, str], Awaitable[ReasonedPrediction[Any]]],
        question: MetaculusQuestion,
        prompt: str,
    ) -> ReasonedPrediction[Any]:
        """Run one forecast on the next model in turn; on failure try the next one."""
        last_error: Exception | None = None
        for attempt in range(1, FORECAST_ATTEMPTS + 1):
            llm = next(self._forecast_cycle)
            token = _forecast_llm.set(llm)
            try:
                prediction = await forecast(question, prompt)
                return ReasonedPrediction(
                    prediction_value=prediction.prediction_value,
                    reasoning=f"*Forecast by {llm.model}*\n\n{prediction.reasoning}",
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    f"Forecast attempt {attempt}/{FORECAST_ATTEMPTS} with {llm.model} "
                    f"failed for {question.page_url}: {type(error).__name__}: {error}"
                )
            finally:
                _forecast_llm.reset(token)
        assert last_error is not None
        raise last_error

    async def _binary_prompt_to_forecast(
        self, question: BinaryQuestion, prompt: str
    ) -> ReasonedPrediction[float]:
        prediction = await self._guarded(
            super()._binary_prompt_to_forecast, question, prompt
        )
        return ReasonedPrediction(
            prediction_value=clamp_probability(prediction.prediction_value),
            reasoning=prediction.reasoning,
        )

    async def _multiple_choice_prompt_to_forecast(
        self, question: MultipleChoiceQuestion, prompt: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prediction = await self._guarded(
            super()._multiple_choice_prompt_to_forecast, question, prompt
        )
        options = prediction.prediction_value.predicted_options
        floored = floor_option_probabilities([o.probability for o in options])
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=option.option_name, probability=p)
                    for option, p in zip(options, floored)
                ]
            ),
            reasoning=prediction.reasoning,
        )

    async def _numeric_prompt_to_forecast(
        self, question: NumericQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._guarded(super()._numeric_prompt_to_forecast, question, prompt)

    async def _date_prompt_to_forecast(
        self, question: DateQuestion, prompt: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return await self._guarded(super()._date_prompt_to_forecast, question, prompt)
