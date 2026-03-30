# CreditSense Public Repo

Kaggle competition link: [CreditSense challenge](https://www.kaggle.com/t/3e62a127eb85418aa851a5ee258e7c04)

This repo keeps the best public submission lane from the project in a cleaner form. The main report focuses on the final public solution, while the wider project history is documented separately in the methods report.

## What is here

- `notebooks/01_best_public_pipeline.ipynb` runs the public-best pipeline.
- `notebooks/02_data_exploration.ipynb` regenerates the paper figures.
- `notebooks/03_solution_analysis.ipynb` shows the score progression and feature signal comparisons.
- `creditsense_public/` contains the helper modules used by the notebooks.
- `artifacts/overnight/` stores the real overnight tuning outputs and validation summaries copied from the working project.
- `artifacts/report/` stores the small CSV tables used for report figures.
- `artifacts/history/` stores the compact historical score tables used by the detailed methods report.
- `paper/main.tex` is the LaTeX source for the main report.
- `paper/main.pdf` is the compiled main report.
- `methods_tried_detailed_report/main.tex` is the LaTeX source for the longer timeline report.
- `methods_tried_detailed_report/main.pdf` is the compiled timeline report.

## Setup

1. Create a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Place `credit_train.csv` and `credit_test.csv` in one of these locations:
   - the repo root
   - the parent folder of the repo
   - the grandparent folder of the repo

The current workspace already keeps the CSV files one level above the repo, so the notebooks can find them automatically here.

## Running the notebooks

- Open `notebooks/01_best_public_pipeline.ipynb` for the main pipeline.
- The committed notebook output uses `RUN_SMOKE_TEST = True` so it stays fast enough for a smoke check.
- Switch `RUN_SMOKE_TEST` to `False` if you want the heavier public-best run.
- Open `notebooks/02_data_exploration.ipynb` to regenerate the PNG and HTML figures under `paper/figures/`.
- Open `notebooks/03_solution_analysis.ipynb` to inspect the score tables and feature comparisons used in the report.
- The methods report figures can be rebuilt with the helper module if you do not want to open a notebook:

```bash
python - <<'PY'
from pathlib import Path
from creditsense_public.reporting import build_methods_report_figures
build_methods_report_figures(Path("methods_tried_detailed_report/figures"))
PY
```

## Rebuilding the reports

If you have `tectonic` and `pdftoppm` installed:

```bash
tectonic --keep-logs --keep-intermediates --outdir paper paper/main.tex
pdftoppm -png paper/main.pdf paper/_rendered/main
tectonic --keep-logs --keep-intermediates --outdir methods_tried_detailed_report methods_tried_detailed_report/main.tex
pdftoppm -png methods_tried_detailed_report/main.pdf methods_tried_detailed_report/_rendered/main
```

The `pdftoppm` commands export page previews so you can visually check the PDFs instead of trusting the source.

## Notes

- The public notebook now reads the original overnight tuning artifacts from `artifacts/overnight/` instead of hiding scores and settings in a separate leaderboard-style Python file.
- The final public submission path in this repo is `meta_overnight`.
- The strongest later local score was higher, but it came from a heavier local-only blend that is covered in the separate methods report rather than the main report.
- The main report explains the clean public submission lane. The longer project history, branch timeline, and “what we tried” story live in `methods_tried_detailed_report/main.pdf`.
