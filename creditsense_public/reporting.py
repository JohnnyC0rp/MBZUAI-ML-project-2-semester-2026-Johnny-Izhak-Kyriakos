from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.feature_selection import f_classif, f_regression

from .features import add_features, load_competition_data
from .historical_results import FEATURE_ENGINEERING_STAGES, MODEL_LEADERBOARD, STAGE_PROGRESS


def _write_plotly(fig, stem: str, figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    html_path = figure_dir / "interactive" / f"{stem}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(png_path, scale=2)
    fig.write_html(html_path, include_plotlyjs="cdn")
    return {"png": png_path, "html": html_path}


def build_report_figures(figure_dir: Path) -> dict[str, dict[str, Path]]:
    data = load_competition_data()
    train_df = data["train_df"]
    engineered = add_features(train_df.drop(columns=["RiskTier", "InterestRate"]))

    outputs: dict[str, dict[str, Path]] = {}

    class_counts = train_df["RiskTier"].value_counts().sort_index().rename_axis("RiskTier").reset_index(name="Applicants")
    fig = px.bar(
        class_counts,
        x="RiskTier",
        y="Applicants",
        title="RiskTier distribution in the training set",
        color="RiskTier",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(showlegend=False)
    outputs["class_distribution"] = _write_plotly(fig, "class_distribution", figure_dir)

    missing = (
        train_df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .rename("missing_share")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing["missing_pct"] = missing["missing_share"] * 100
    fig = px.bar(
        missing,
        x="missing_pct",
        y="feature",
        orientation="h",
        title="Top missing-value rates",
        color="missing_pct",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    outputs["missingness"] = _write_plotly(fig, "missingness_overview", figure_dir)

    stage_df = pd.DataFrame(STAGE_PROGRESS)
    fig = px.line(
        stage_df,
        x="stage",
        y="combined",
        markers=True,
        title="Validation score progression across the project",
        color_discrete_sequence=["#1f6aa5"],
    )
    fig.update_layout(xaxis_tickangle=-25, showlegend=False)
    outputs["stage_progress"] = _write_plotly(fig, "stage_progress", figure_dir)

    leaderboard_df = pd.DataFrame(MODEL_LEADERBOARD)
    fig = px.bar(
        leaderboard_df.sort_values("combined", ascending=True),
        x="combined",
        y="model",
        orientation="h",
        title="Representative model leaderboard",
        color="family",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    outputs["leaderboard"] = _write_plotly(fig, "model_leaderboard", figure_dir)

    feature_stage_df = pd.DataFrame(FEATURE_ENGINEERING_STAGES)
    fig = px.bar(
        feature_stage_df,
        x="addition",
        y="validation_combined",
        title="Validation score after major feature-and-model additions",
        color="validation_combined",
        color_continuous_scale="Mint",
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    outputs["feature_steps"] = _write_plotly(fig, "feature_stage_progress", figure_dir)

    numeric_engineered = engineered.select_dtypes(include=[np.number]).fillna(0.0)
    cls_scores, _ = f_classif(numeric_engineered, train_df["RiskTier"].to_numpy())
    reg_scores, _ = f_regression(numeric_engineered, train_df["InterestRate"].to_numpy())
    score_df = pd.DataFrame(
        {
            "feature": numeric_engineered.columns,
            "classification_signal": np.nan_to_num(cls_scores, nan=0.0, posinf=0.0, neginf=0.0),
            "regression_signal": np.nan_to_num(reg_scores, nan=0.0, posinf=0.0, neginf=0.0),
        }
    )
    top_cls = score_df.nlargest(10, "classification_signal")[["feature", "classification_signal"]].assign(task="Classification")
    top_reg = score_df.nlargest(10, "regression_signal")[["feature", "regression_signal"]].assign(task="Regression")
    top_cls = top_cls.rename(columns={"classification_signal": "signal"})
    top_reg = top_reg.rename(columns={"regression_signal": "signal"})
    fig = px.bar(
        pd.concat([top_cls, top_reg], ignore_index=True),
        x="signal",
        y="feature",
        color="task",
        facet_col="task",
        orientation="h",
        title="Top engineered features for classification vs regression",
        color_discrete_sequence=["#0f8b8d", "#b83b5e"],
    )
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=")[-1]))
    outputs["feature_signal"] = _write_plotly(fig, "feature_signal_comparison", figure_dir)

    risk_rate = (
        train_df.groupby("RiskTier", as_index=False)["InterestRate"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "MeanAPR", "median": "MedianAPR"})
    )
    fig = px.bar(
        risk_rate,
        x="RiskTier",
        y=["MeanAPR", "MedianAPR"],
        barmode="group",
        title="Interest rate rises with predicted risk tier",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    outputs["rate_by_risk"] = _write_plotly(fig, "rate_by_risk", figure_dir)

    return outputs
