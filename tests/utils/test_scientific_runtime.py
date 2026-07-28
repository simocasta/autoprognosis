from __future__ import annotations

import inspect

from autoprognosis.explorers.core.optimizers.bayesian import BayesianOptimizer
from autoprognosis.plugins.prediction.classifiers import plugin_catboost, plugin_lgbm


def test_bayesian_studies_are_fresh_and_process_local_by_default() -> None:
    signature = inspect.signature(BayesianOptimizer.create_study)
    assert signature.parameters["load_if_exists"].default is False
    assert signature.parameters["storage_type"].default == "none"


def test_catboost_respects_learner_thread_budget(monkeypatch) -> None:
    monkeypatch.setattr(plugin_catboost, "n_learner_jobs", lambda: 3)
    plugin = plugin_catboost.CatBoostPlugin(n_estimators=2)
    assert plugin.model.get_params()["thread_count"] == 3


def test_lightgbm_respects_learner_thread_budget(monkeypatch) -> None:
    monkeypatch.setattr(plugin_lgbm, "n_learner_jobs", lambda: 5)
    plugin = plugin_lgbm.LightGBMPlugin(n_estimators=2, calibration=0)
    assert plugin.model.get_params()["n_jobs"] == 5
