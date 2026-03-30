from __future__ import annotations


KAGGLE_COMPETITION_URL = "https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04"

PUBLIC_BEST_MODEL = "meta_overnight"
PUBLIC_BEST_LOCAL_COMBINED = 0.8494888746057239
PRIVATE_BEST_LOCAL_COMBINED = 0.857645444869995

PUBLIC_BLEND_WEIGHTS = {
    "classification": {
        "xgb": 0.008740858016119181,
        "lgb": 0.20307576518618675,
        "cat": 0.11571774095618016,
        "et": 0.0005104289674535134,
        "mps": 0.6719552068740603,
    },
    "regression": {
        "xgb": 0.34518176535858996,
        "lgb": 0.0007090315461353092,
        "cat": 0.6525907505844593,
        "et": 0.0008418488620459765,
        "mps": 0.0006766036487695478,
    },
}

PUBLIC_META_PARAMS = {
    "classifier_C": 2.382599395818209,
    "regressor_alpha": 59.71871840227128,
}

STAGE_PROGRESS = [
    {
        "stage": "Assignment baseline",
        "combined": 0.5100000000000000,
        "category": "baseline",
        "summary": "Template logistic-regression and linear-regression workflow from the assignment PDF.",
    },
    {
        "stage": "Polynomial parallel blend",
        "combined": 0.8225062114511217,
        "category": "feature_engineering",
        "summary": "Shared financial ratios, semantic missingness flags, ordinal encoding, and controlled polynomial interactions.",
    },
    {
        "stage": "Boosted overnight meta ensemble",
        "combined": 0.8494888746057239,
        "category": "public_best",
        "summary": "XGBoost, LightGBM, CatBoost, ExtraTrees, and an MPS neural branch combined with a compact meta-model.",
    },
    {
        "stage": "Target-stat bagging",
        "combined": 0.8557313795770918,
        "category": "local_research",
        "summary": "OOF target encoding and calibration-heavy tree blending pushed the local score above the public lane.",
    },
    {
        "stage": "MLP meta stack",
        "combined": 0.8571810142312731,
        "category": "local_research",
        "summary": "A compact tabular MLP added a new lane on top of the target-stat family.",
    },
    {
        "stage": "Final weighted local blend",
        "combined": 0.8576454448699950,
        "category": "local_best",
        "summary": "The strongest local score used a tiny-weight historical blend on top of the MLP and target-stat lanes.",
    },
]

FEATURE_ENGINEERING_STAGES = [
    {
        "addition": "Domain ratios and semantic missingness flags",
        "validation_combined": 0.8225062114511217,
        "why_it_helped": "Banks care about affordability, delinquency severity, and why values are missing, not just whether they are.",
    },
    {
        "addition": "Controlled polynomial interactions",
        "validation_combined": 0.8251399274553571,
        "why_it_helped": "Pairwise stress interactions let tree models capture non-linear risk cliffs without exploding the feature space into chaos.",
    },
    {
        "addition": "Tree ensemble diversification",
        "validation_combined": 0.8494888746057239,
        "why_it_helped": "XGBoost, LightGBM, CatBoost, and ExtraTrees made different mistakes, which is ensemble catnip.",
    },
    {
        "addition": "Target statistics and calibrated bagging",
        "validation_combined": 0.8557313795770918,
        "why_it_helped": "OOF target encodings injected supervised structure into sparse categorical groups while keeping validation leakage under control.",
    },
    {
        "addition": "Neural branch and weighted fusion",
        "validation_combined": 0.8576454448699950,
        "why_it_helped": "A compact MLP captured smooth non-linear effects that complemented the tree family in the final blend.",
    },
]

MODEL_LEADERBOARD = [
    {"model": "meta_overnight", "accuracy": 0.8477142857142858, "r2": 0.8512634634971619, "combined": 0.8494888746057239, "family": "public_best"},
    {"model": "blend_overnight", "accuracy": 0.8424285714285714, "r2": 0.8485081791877747, "combined": 0.8454683753081731, "family": "public_best"},
    {"model": "target_stats_blend", "accuracy": 0.8562857142857143, "r2": 0.8551770448684692, "combined": 0.8557313795770918, "family": "local_research"},
    {"model": "mlp_meta_stack_v1", "accuracy": 0.8587142857142858, "r2": 0.8556477427482605, "combined": 0.8571810142312731, "family": "local_research"},
    {"model": "mlp_weighted_blend_v1", "accuracy": 0.8600000000000000, "r2": 0.8552908897399902, "combined": 0.8576454448699950, "family": "local_best"},
]
