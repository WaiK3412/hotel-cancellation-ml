# Hotel booking cancellation prediction

A machine learning consultancy project (6006CEM group coursework). We rank hotel bookings by their risk of cancellation using only information known at booking time. We compare supervised and unsupervised learning, test every preparation step, tune three model families and combine the best two. The recommended model is **soft voting of Random Forest, Hist Gradient Boosting**: test AP **0.698** against a baseline of 0.271.

Group: [GROUP TO FILL: member names and student IDs]

## What is here

The technical report and presentation slides are submitted on Canvas. This repository holds the work behind them: the executed Jupyter notebooks, the source code and the cleaned dataset. Every notebook is saved with its outputs, so all tables and charts can be read without running anything.

## Dataset

Antonio, de Almeida and Nunes (2019), *Hotel booking demand datasets*, *Data in Brief* 22, 41–49, [doi:10.1016/j.dib.2018.11.126](https://doi.org/10.1016/j.dib.2018.11.126). The licence is CC BY 4.0. The CSV is downloaded from [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-02-11) by notebook 00 and is not committed; its SHA-256 is recorded in `data/processed/split_manifest.json`.

- **Size:** 119,390 bookings and 31 fields plus the target. After removing 31,994 exact duplicates, 87,396 remain.
- **Split:** group-aware 80/20 into 69,877 training and 17,519 test bookings (seed 42). Bookings identical on every booking-time field always stay on the same side.
- **Excluded as leakage:** country, required_car_parking_spaces, deposit_type, arrival_date_year. Their values are recorded at or after check-in; see Section 4.1 of the report.

## Repository map

| Path | Contents |
|---|---|
| `notebooks/` | Executed Jupyter notebooks: 00–04 are the core Parts A–H; 05–10 are robustness extensions |
| `src/` | Source code the notebooks call: data, features, eda, preprocess, models, evaluate, robustness, nested, clustering, decision, interpret |
| `scripts/` | Dataset screening behind Appendix E of the report |
| `data/processed/` | Cleaned dataset (`cleaned.csv`), the 80/20 split indices and the split manifest |
| `results/`, `reports/figures/`, `models/` | Empty; the notebooks write their tables, charts and the final model here when run |

## Setup and re-running

Nothing needs to be run to read the results: open any notebook to see its saved tables and charts. To re-run, use Python 3.13 and run every command from the repository root.

~~~powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name ml-project
.venv\Scripts\jupyter lab
~~~

On Linux or macOS use `python3` and `.venv/bin/` instead of `.venv\Scripts\`. Then run the notebooks in order, 00 to 10, with the `ml-project` kernel: each one saves what the next ones need into `results/`. Notebook 00 downloads the raw data, so it needs internet access. A full run takes about five hours on a laptop; notebooks 03, 04 and 06 take about an hour each.

## Results (test set)

| Model                  |    AP |   ROC-AUC |    F1 |   Recall |   Precision |   Accuracy |   Balanced acc. |
|:-----------------------|------:|----------:|------:|---------:|------------:|-----------:|----------------:|
| Majority baseline      | 0.271 |     0.500 | 0.000 |    0.000 |       0.000 |      0.729 |           0.500 |
| Stratified baseline    | 0.270 |     0.498 | 0.268 |    0.269 |       0.267 |      0.603 |           0.498 |
| Logistic Regression    | 0.633 |     0.787 | 0.526 |    0.419 |       0.706 |      0.795 |           0.677 |
| Random Forest          | 0.696 |     0.835 | 0.593 |    0.507 |       0.714 |      0.811 |           0.716 |
| Hist Gradient Boosting | 0.689 |     0.833 | 0.582 |    0.493 |       0.710 |      0.808 |           0.709 |
| Voting                 | 0.698 |     0.838 | 0.591 |    0.501 |       0.721 |      0.812 |           0.714 |
| Stacking               | 0.698 |     0.838 | 0.595 |    0.510 |       0.716 |      0.812 |           0.717 |

AP is average precision; the other columns use a 0.5 cutoff.

- **Tuning:** on the test set, tuning added +0.017 AP for the random forest and +0.013 for gradient boosting, and nothing for logistic regression.
- **Over time:** on bookings made after the training period, AP falls to 0.548.
- **Lead time:** the model catches most cancellations of bookings made months ahead, but only 0.086 of those made within a week of arrival.
