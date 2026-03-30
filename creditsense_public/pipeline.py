from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .artifacts import PUBLIC_BEST_MODEL
from .features import fit_preprocessor, load_competition_data, split_train_validation
from .models import (
    build_submission_frame,
    evaluate_predictions,
    fit_meta_ensemble,
    fit_weighted_blend,
    train_catboost_branch,
    train_extratrees_branch,
    train_lgb_branch,
    train_mps_branch,
    train_xgb_branch,
)


@dataclass
class PipelineConfig:
    smoke_test: bool = False


def _prepare_matrices() -> dict:
    data = load_competition_data()
    X = data["X"]
    y_cls = data["y_cls"]
    y_reg = data["y_reg"]
    test_df = data["test_df"]

    (
        X_train_raw,
        X_val_raw,
        y_train_cls,
        y_val_cls,
        y_train_reg,
        y_val_reg,
    ) = split_train_validation(X, y_cls, y_reg)

    pre_split = fit_preprocessor(X_train_raw, use_poly=True)
    pre_full = fit_preprocessor(X, use_poly=True)

    X_train = pre_split.transform(X_train_raw)
    X_val = pre_split.transform(X_val_raw)
    X_full = pre_full.transform(X)
    X_test = pre_full.transform(test_df)

    cat_train = X_train_raw.copy()
    cat_val = X_val_raw.copy()
    cat_full = X.copy()
    cat_test = test_df.copy()
    for frame, cat_cols in [
        (cat_train, pre_split.cat_cols),
        (cat_val, pre_split.cat_cols),
        (cat_full, pre_full.cat_cols),
        (cat_test, pre_full.cat_cols),
    ]:
        for column in cat_cols:
            if column in frame.columns:
                frame[column] = frame[column].fillna("__MISSING__").astype(str)

    return {
        "X_train_df": X_train,
        "X_val_df": X_val,
        "X_full_df": X_full,
        "X_test_df": X_test,
        "X_train_num": X_train.values.astype("float32"),
        "X_val_num": X_val.values.astype("float32"),
        "X_full_num": X_full.values.astype("float32"),
        "X_test_num": X_test.values.astype("float32"),
        "X_train_cat": cat_train,
        "X_val_cat": cat_val,
        "X_full_cat": cat_full,
        "X_test_cat": cat_test,
        "cat_columns": pre_full.cat_cols,
        "feature_columns": X_train.columns.tolist(),
        "y_train_cls": y_train_cls.to_numpy(dtype="int64"),
        "y_val_cls": y_val_cls.to_numpy(dtype="int64"),
        "y_train_reg": y_train_reg.to_numpy(dtype="float32"),
        "y_val_reg": y_val_reg.to_numpy(dtype="float32"),
        "y_cls": y_cls.to_numpy(dtype="int64"),
        "y_reg": y_reg.to_numpy(dtype="float32"),
    }


def run_public_best_pipeline(config: PipelineConfig | None = None) -> dict:
    config = PipelineConfig() if config is None else config
    matrices = _prepare_matrices()

    branches = []
    branches.append(
        train_xgb_branch(
            matrices["X_train_num"],
            matrices["X_val_num"],
            matrices["X_full_num"],
            matrices["X_test_num"],
            matrices["y_train_cls"],
            matrices["y_train_reg"],
            matrices["y_cls"],
            matrices["y_reg"],
            smoke_test=config.smoke_test,
        )
    )
    branches.append(
        train_lgb_branch(
            matrices["X_train_df"],
            matrices["X_val_df"],
            matrices["X_full_df"],
            matrices["X_test_df"],
            matrices["y_train_cls"],
            matrices["y_train_reg"],
            matrices["y_cls"],
            matrices["y_reg"],
            smoke_test=config.smoke_test,
        )
    )
    branches.append(
        train_catboost_branch(
            matrices["X_train_cat"],
            matrices["X_val_cat"],
            matrices["X_full_cat"],
            matrices["X_test_cat"],
            matrices["cat_columns"],
            matrices["y_train_cls"],
            matrices["y_train_reg"],
            matrices["y_cls"],
            matrices["y_reg"],
            smoke_test=config.smoke_test,
        )
    )
    branches.append(
        train_extratrees_branch(
            matrices["X_train_num"],
            matrices["X_val_num"],
            matrices["X_full_num"],
            matrices["X_test_num"],
            matrices["y_train_cls"],
            matrices["y_train_reg"],
            matrices["y_cls"],
            matrices["y_reg"],
            smoke_test=config.smoke_test,
        )
    )
    branches.append(
        train_mps_branch(
            matrices["X_train_num"],
            matrices["X_val_num"],
            matrices["X_full_num"],
            matrices["X_test_num"],
            matrices["y_train_cls"],
            matrices["y_val_cls"],
            matrices["y_train_reg"],
            matrices["y_val_reg"],
            matrices["y_cls"],
            matrices["y_reg"],
            smoke_test=config.smoke_test,
        )
    )

    blend = fit_weighted_blend(
        branches,
        matrices["y_val_cls"],
        matrices["y_val_reg"],
        smoke_test=config.smoke_test,
    )
    meta = fit_meta_ensemble(
        branches,
        matrices["y_val_cls"],
        matrices["y_val_reg"],
        smoke_test=config.smoke_test,
    )

    score_rows = []
    for branch in [*branches, blend, meta]:
        metrics = evaluate_predictions(
            matrices["y_val_cls"],
            matrices["y_val_reg"],
            branch.val_proba,
            branch.val_reg,
        )
        score_rows.append(
            {
                "model": branch.name,
                "accuracy": metrics.accuracy,
                "r2": metrics.r2,
                "combined": metrics.combined,
            }
        )

    score_table = pd.DataFrame(score_rows).sort_values("combined", ascending=False).reset_index(drop=True)
    branch_map = {branch.name: branch for branch in [*branches, blend, meta]}
    best_predictions = branch_map[PUBLIC_BEST_MODEL]

    return {
        "config": config,
        "score_table": score_table,
        "submission": build_submission_frame(best_predictions),
        "branches": branch_map,
        "feature_columns": matrices["feature_columns"],
        "cat_columns": matrices["cat_columns"],
        "y_val_cls": matrices["y_val_cls"],
        "y_val_reg": matrices["y_val_reg"],
        "X_train_num": matrices["X_train_num"],
        "X_val_num": matrices["X_val_num"],
    }
