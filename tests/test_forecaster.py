import asyncio
import os
import unittest
from unittest import mock

os.environ.setdefault("METACULUS_TOKEN", "test-token")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from forecasting_tools import PredictedOptionList, ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from bot import settings as settings_module
from bot.forecaster import TournamentBot
from main import FallTemplateBot2026


class FakeQuestion:
    page_url = "https://example.com/q/1"


def make_bot() -> TournamentBot:
    with mock.patch.dict(os.environ, {"BOT_PROFILE": "free"}):
        return TournamentBot(settings_module.resolve_settings())


class SettingsTest(unittest.TestCase):
    def test_free_profile_uses_only_free_models(self):
        with mock.patch.dict(os.environ, {"BOT_PROFILE": "free"}):
            settings = settings_module.resolve_settings()
        self.assertEqual(settings.profile, "free")
        self.assertTrue(all(name.endswith(":free") for name in settings.forecast_model_names))
        self.assertTrue(settings.utility_llm.model.endswith(":free"))
        self.assertIsNone(settings.search_llm)

    def test_paid_profile_downgrades_and_stops_with_credits(self):
        env = {"BOT_PROFILE": "paid"}
        with mock.patch.dict(os.environ, env), mock.patch.object(settings_module, "fetch_openrouter_credits", return_value=50.0):
            full = settings_module.resolve_settings()
        with mock.patch.dict(os.environ, env), mock.patch.object(settings_module, "fetch_openrouter_credits", return_value=3.0):
            low = settings_module.resolve_settings()
        with mock.patch.dict(os.environ, env), mock.patch.object(settings_module, "fetch_openrouter_credits", return_value=0.1):
            empty = settings_module.resolve_settings()
        self.assertEqual(full.forecast_model_names, settings_module.PAID_FORECAST_MODELS)
        self.assertIsNotNone(full.search_llm)
        self.assertEqual(low.forecast_model_names, settings_module.PAID_CHEAP_FORECAST_MODELS)
        self.assertIsNone(low.search_llm)
        self.assertIsNotNone(empty.skip_reason)

    def test_missing_key_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaises(SystemExit):
                settings_module.resolve_settings()


class GuardedForecastTest(unittest.TestCase):
    def test_concurrent_forecasts_each_get_their_own_model(self):
        bot = make_bot()
        seen: list[tuple[str, str]] = []

        async def forecast(question, prompt):
            before = bot.get_llm("default", "llm").model
            await asyncio.sleep(0.01)  # let the other forecasts start
            seen.append((before, bot.get_llm("default", "llm").model))
            return ReasonedPrediction(prediction_value=0.5, reasoning="because")

        async def run():
            return await asyncio.gather(*(bot._guarded(forecast, FakeQuestion(), "p") for _ in range(3)))

        predictions = asyncio.run(run())
        self.assertTrue(all(before == after for before, after in seen))
        self.assertEqual(sorted(before for before, _ in seen), sorted(bot.settings.forecast_model_names))
        self.assertTrue(all(p.reasoning.startswith("*Forecast by openrouter/") for p in predictions))
        # Outside a forecast the configured default is returned again.
        self.assertEqual(bot.get_llm("default", "llm").model, bot.settings.forecast_model_names[0])

    def test_failure_is_retried_on_the_next_model(self):
        bot = make_bot()
        used: list[str] = []

        async def forecast(question, prompt):
            used.append(bot.get_llm("default", "llm").model)
            if len(used) == 1:
                raise RuntimeError("provider down")
            return ReasonedPrediction(prediction_value=0.4, reasoning="ok")

        prediction = asyncio.run(bot._guarded(forecast, FakeQuestion(), "p"))
        self.assertEqual(prediction.prediction_value, 0.4)
        self.assertEqual(len(used), 2)
        self.assertNotEqual(used[0], used[1])

    def test_gives_up_after_all_attempts(self):
        bot = make_bot()

        async def forecast(question, prompt):
            raise RuntimeError("still down")

        with self.assertRaises(RuntimeError):
            asyncio.run(bot._guarded(forecast, FakeQuestion(), "p"))


class PostProcessingTest(unittest.TestCase):
    def test_binary_forecast_is_clamped(self):
        bot = make_bot()

        async def overconfident(self, question, prompt):
            return ReasonedPrediction(prediction_value=0.99, reasoning="sure")

        with mock.patch.object(FallTemplateBot2026, "_binary_prompt_to_forecast", overconfident):
            prediction = asyncio.run(bot._binary_prompt_to_forecast(FakeQuestion(), "p"))
        self.assertEqual(prediction.prediction_value, 0.98)

    def test_multiple_choice_options_get_a_floor(self):
        bot = make_bot()

        async def lopsided(self, question, prompt):
            options = [PredictedOption(option_name="A", probability=0.999), PredictedOption(option_name="B", probability=0.001)]
            return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=options), reasoning="sure")

        with mock.patch.object(FallTemplateBot2026, "_multiple_choice_prompt_to_forecast", lopsided):
            prediction = asyncio.run(bot._multiple_choice_prompt_to_forecast(FakeQuestion(), "p"))
        probabilities = {o.option_name: o.probability for o in prediction.prediction_value.predicted_options}
        self.assertAlmostEqual(probabilities["B"], 0.01, places=3)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
