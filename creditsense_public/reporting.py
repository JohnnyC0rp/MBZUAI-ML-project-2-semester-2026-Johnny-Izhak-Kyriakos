from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import f_classif, f_regression

from .artifacts import load_history_table, load_report_table
from .features import add_features, load_competition_data


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

    stage_df = load_report_table("stage_progress.csv")
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

    leaderboard_df = load_report_table("model_leaderboard.csv")
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

    feature_stage_df = load_report_table("feature_engineering_stages.csv")
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


def build_methods_report_figures(figure_dir: Path) -> dict[str, dict[str, Path]]:
    history_df = load_history_table("approach_chronology.csv").copy()
    history_df["date"] = pd.to_datetime(history_df["date"])
    history_df = history_df.sort_values("date").reset_index(drop=True)
    history_df["best_so_far"] = history_df["combined"].cummax()

    outputs: dict[str, dict[str, Path]] = {}

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df["combined"],
            mode="markers",
            name="Saved best of each run",
            marker=dict(size=8, color="#7aa5c7", line=dict(width=1, color="white")),
            customdata=np.stack(
                [
                    history_df["best_model"],
                    history_df["family"],
                    history_df["compute_lane"],
                    history_df["accuracy"],
                    history_df["r2"],
                    history_df["elapsed_minutes"].fillna(-1.0),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Family: %{customdata[1]}<br>"
                "Compute: %{customdata[2]}<br>"
                "Accuracy: %{customdata[3]:.4f}<br>"
                "R2: %{customdata[4]:.4f}<br>"
                "Elapsed minutes: %{customdata[5]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df["best_so_far"],
            mode="lines+markers",
            name="Best score so far",
            line=dict(color="#0f4c5c", width=3),
            marker=dict(size=6, color="#0f4c5c"),
            hovertemplate="Date: %{x|%b %d}<br>Best score so far: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Saved best-score progression across the project",
        legend_title_text="",
        xaxis_title="",
        yaxis_title="Combined validation score",
        yaxis_range=[0.82, 0.86],
    )
    outputs["score_progression"] = _write_plotly(fig, "methods_score_progression", figure_dir)

    top_df = history_df.nlargest(10, "combined").sort_values("combined", ascending=True)
    fig = px.bar(
        top_df,
        x="combined",
        y="best_model",
        orientation="h",
        color="family",
        text="combined",
        title="Top saved approaches by combined validation score",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="inside")
    fig.update_layout(legend_title_text="", xaxis_title="Combined validation score", yaxis_title="", xaxis_range=[0, 0.9])
    outputs["top_approaches"] = _write_plotly(fig, "methods_top_approaches", figure_dir)

    runtime_df = history_df.dropna(subset=["elapsed_minutes"]).copy()
    fig = px.scatter(
        runtime_df,
        x="elapsed_minutes",
        y="combined",
        color="compute_lane",
        hover_data={"family": True, "best_model": True},
        hover_name="best_model",
        title="Runtime versus score for the saved experiment runs",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(legend_title_text="", xaxis_title="Elapsed minutes", yaxis_title="Combined validation score")
    outputs["runtime_tradeoff"] = _write_plotly(fig, "methods_runtime_tradeoff", figure_dir)

    milestone_df = load_history_table("timeline_milestones.csv").copy()
    milestone_df["date_label"] = pd.to_datetime(milestone_df["date"]).dt.strftime("%b %d")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=milestone_df["date_label"],
            y=milestone_df["score_anchor"],
            mode="lines+markers",
            line=dict(color="#0f4c5c", width=3),
            marker=dict(size=12, color="#0f766e", line=dict(width=1, color="white")),
            customdata=np.stack([milestone_df["phase"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>Score anchor: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Main project turns and the score anchors they produced",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Score anchor",
        yaxis_range=[0.818, 0.86],
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=milestone_df["date_label"].tolist(),
        tickangle=-20,
    )
    outputs["milestones"] = _write_plotly(fig, "methods_milestones", figure_dir)

    public_df = load_history_table("public_checks.csv").copy()
    fig = px.bar(
        public_df,
        x="submission",
        y="public_score",
        color="submission",
        title="First public Kaggle calibration checks that were explicitly recorded",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Public Kaggle score")
    outputs["public_checks"] = _write_plotly(fig, "methods_public_checks", figure_dir)

    return outputs
