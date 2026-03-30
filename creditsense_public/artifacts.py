from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .paths import repo_root


KAGGLE_COMPETITION_URL = "https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04"
PUBLIC_BEST_MODEL = "meta_overnight"


def artifact_root() -> Path:
    return repo_root() / "artifacts"


def overnight_artifact(name: str) -> Path:
    return artifact_root() / "overnight" / name


def report_artifact(name: str) -> Path:
    return artifact_root() / "report" / name


def history_artifact(name: str) -> Path:
    return artifact_root() / "history" / name


@lru_cache(maxsize=None)
def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_overnight_branch_settings() -> dict[str, dict]:
    xgb = load_json(overnight_artifact("xgb_tuning.json"))
    lgb = load_json(overnight_artifact("lgb_tuning.json"))
    cat = load_json(overnight_artifact("catboost_tuning.json"))
    et = load_json(overnight_artifact("extratrees.json"))
    mps = load_json(overnight_artifact("mps_ensemble.json"))
    return {
        "xgb": {
            "classifier": xgb["best_params"]["xgb_classifier"],
            "regressor": xgb["best_params"]["xgb_regressor"],
        },
        "lgb": {
            "classifier": lgb["best_params"]["lgbm_classifier"],
            "regressor": lgb["best_params"]["lgbm_regressor"],
        },
        "cat": {
            "classifier": cat["best_params"]["catboost_classifier"],
            "regressor": cat["best_params"]["catboost_regressor"],
        },
        "et": {
            "n_estimators_classifier": et["n_estimators_classifier"],
            "n_estimators_regressor": et["n_estimators_regressor"],
        },
        "mps": mps,
    }


@lru_cache(maxsize=1)
def load_overnight_search_budgets() -> dict:
    return load_json(overnight_artifact("overnight_results.json"))


def load_report_table(name: str) -> pd.DataFrame:
    return load_csv(report_artifact(name))


def load_history_table(name: str) -> pd.DataFrame:
    return load_csv(history_artifact(name))
