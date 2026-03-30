# CreditSense Public Report Repo

Kaggle competition link: [CreditSense challenge](https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04)

This public repo packages the strongest **report-ready Kaggle pipeline** from the project: `meta_overnight`, a fixed-parameter ensemble of XGBoost, LightGBM, CatBoost, ExtraTrees, and a compact MPS neural branch. The accompanying paper explains the exploratory findings, why semantic missingness and financial-ratio features mattered, how the ensemble evolved from the assignment baseline to the public-best pipeline, and how later local-only stacking pushed validation slightly higher than the public lane.

Main entry points:

- [01_best_public_pipeline.ipynb](notebooks/01_best_public_pipeline.ipynb) runs the clean public-best pipeline.
- [02_data_exploration.ipynb](notebooks/02_data_exploration.ipynb) regenerates the report figures.
- [03_solution_analysis.ipynb](notebooks/03_solution_analysis.ipynb) summarizes score progression and feature signal differences.
- [main.tex](paper/main.tex) is the LaTeX source for the report.
- [main.pdf](paper/main.pdf) is the compiled paper.

To run the notebooks, place `credit_train.csv` and `credit_test.csv` in the repo root or one of its parent folders. The committed notebook output uses smoke-test mode for fast verification; switch `RUN_SMOKE_TEST` to `False` inside the pipeline notebook for the full public-best run.
