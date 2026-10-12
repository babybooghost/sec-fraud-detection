# SEC Fraud Detection

A research pipeline for accounting fraud classification using annual SEC financial statements and Bao et al. AAER labels.

It builds Beneish ratios and nonlinear features, trains XGBoost on historical filings, and checks performance and attribution drift on forward years. Python 3.11+.

## Browser demo

[Open the dashboard](https://babybooghost.github.io/sec-fraud-detection/). No terminal, account, or API key is needed.

The dashboard shows actual Python model outputs from **synthetic inputs**. Change the decision threshold, inspect errors, and download the results. Browser controls recalculate decisions from saved scores; they do not retrain models. All displayed metrics describe this generated experiment, not real-world performance.

To regenerate the checked-in dataset after installing dependencies:

```sh
.venv/bin/python build_demo.py
.venv/bin/python -m http.server 8765 --directory docs
```

Open `http://localhost:8765`. GitHub Pages serves `docs/` from `main`. Commit regenerated `docs/data.json` to publish new results. The dataset tests independently recompute ROC-AUC, average precision, and ROC points from the saved predictions.

## Run

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python pipeline.py \
  --financials data/statements.csv --aaer data/AAER_firm_year.csv \
  --train-end-year 2018 --test-start-year 2019 --test-end-year 2022
.venv/bin/python -m pytest
```

`requirements.lock` pins the tested runtime and test tools. GitHub Actions runs the tests on Python 3.11 and 3.12. Alternatively, install `.[dev]` with pip in an activated environment. On macOS, XGBoost may require the Homebrew `libomp` runtime.

## Input contract

`statements.csv` contains one original annual filing per `(cik, fiscal_year)`. Required fields:

| Field | Definition |
| --- | --- |
| `cik` | Numeric SEC identifier, normalized to ten digits |
| `fiscal_year` | Integer fiscal year |
| `filing_date` | Actual public filing timestamp, parsed as UTC |
| `revenue`, `receivables`, `cogs` | Annual revenue, net receivables, cost of goods sold |
| `current_assets`, `ppe_net`, `total_assets` | Balance sheet amounts, PPE must be net |
| `depreciation`, `sga` | Annual depreciation and selling/general/administrative expenses |
| `current_liabilities`, `long_term_debt` | Balance sheet debt inputs |
| `income_continuing`, `operating_cash_flow` | Income from continuing operations and operating cash flow |

Use consistent currency and scale. Numeric missing values are allowed. Malformed values, infinities, duplicate firm-years, and missing filing dates fail validation. Preserve original filing vintages upstream. The CSV adapter consumes normalized SEC statement exports, not raw Company Facts JSON or EDGAR bulk ZIPs. It does not download filings, resolve XBRL tags, aggregate quarterly facts, or silently substitute net income for continuing income.

AAER input accepts the actual Bao release columns `CIK,YEARA` or canonical `cik,fiscal_year`. Rows are positive observations unless an explicit binary `fraud` field is supplied. Repeated releases collapse by maximum label. Unmatched statements mean no observed AAER event, not verified absence of fraud. Both understatement and overstatement records are included. Filter the source explicitly if the experiment requires a narrower definition.

The Bao modeling CSV is not a substitute for this statement schema. Its listed raw fields do not supply every input needed here, including net PPE, SGA, and operating cash flow. Join a verified SEC/Compustat source before running, and preserve identifier crosswalks.

## Features

Let `t` be the current annual statement and `p` the immediately preceding fiscal year:

- DSRI: `(receivables_t / revenue_t) / (receivables_p / revenue_p)`.
- GMI: `gross_margin_p / gross_margin_t`, where gross margin is `(revenue - cogs) / revenue`.
- AQI: `[1 - (current_assets_t + ppe_net_t) / assets_t] / [1 - (current_assets_p + ppe_net_p) / assets_p]`.
- SGI: `revenue_t / revenue_p`.
- DEPI: `[depreciation_p / (depreciation_p + ppe_net_p)] / [depreciation_t / (depreciation_t + ppe_net_t)]`.
- SGAI: `(sga_t / revenue_t) / (sga_p / revenue_p)`.
- LVGI: `[(current_liabilities_t + long_term_debt_t) / assets_t] / [(current_liabilities_p + long_term_debt_p) / assets_p]`.
- TATA: `(income_continuing_t - operating_cash_flow_t) / assets_t`, the cash-flow convention.

Interactions capture receivables versus sales growth, margin versus growth, asset quality versus accruals, leverage versus accruals, and squared accrual/growth terms. Undefined ratios remain NaN for XGBoost's native missing-value routing. No backward fills. No cross-firm lags. No skipped-year comparisons. A prior filing published after the current filing is ineligible.

## Temporal and cost controls

Training fiscal years end at `train_end_year`, and training filings must be public before January 1 of the next calendar year. This excludes many statements for the final fiscal year because publication happens later. OOT fiscal years begin at `test_start_year`; their filings must arrive after that cutoff. Optional `test_end_year` limits fiscal years, not publication dates. Rows without consecutive annual history are excluded.

All feature transformations are causal. No early stopping, hyperparameter search, resampling, or threshold selection uses OOT data. The same firm may appear in both periods; this is a temporal evaluation, not an unseen-company evaluation.

AAER outcomes are retrospective. The source firm-year file lacks enforcement availability dates. This experiment therefore does not claim historically observable training labels or resolve enforcement-delay/serial-fraud leakage. A deployment backtest needs label-vintage data and a separate label-availability policy. Likewise, upstream restated accounts can defeat point-in-time validity even when this split is correct.

XGBoost uses weighted binary logistic loss with `scale_pos_weight = 10`. This is a differentiable cost-sensitive surrogate, not direct optimization of the discontinuous confusion-matrix utility. Predictions use a fixed 0.5 threshold on weighted model scores. These scores are not calibrated probabilities. Applying a second 1/11 threshold would count the asymmetric penalty twice.

Evaluation reports precision, recall, ROC-AUC, all four confusion counts, total cost `10 * FN + FP`, and utility `-(10 * FN + FP) / N`. Single-class OOT AUC is null.

## SHAP audit

One training-only background explains historical and forward rows in raw log-odds units. The top five drivers are ranked by absolute signed mean SHAP shift. Signed shifts sum to the change in mean prediction margin for the audited samples. Mean absolute attribution changes and PSI also expose distribution changes that cancel in the mean. PSI bins come only from historical SHAP values. These are descriptive diagnostics, not causal claims or significance tests. Deterministic sampling caps audit cost.

## Validation and sources

The 32 tests check hand-calculated ratios, missing denominators, entity/year boundaries, filing availability, label joins, exact asymmetric costs, future-data perturbation invariance, actual XGBoost/SHAP execution, and the CSV CLI. Synthetic scores are not real-data research results.

- [Bao et al. data and reproduction repository](https://github.com/JarFraud/FraudDetection), including the native AAER schema and dataset limitations.
- [Bao et al. (2020), Journal of Accounting Research](https://onlinelibrary.wiley.com/doi/abs/10.1111/1475-679X.12292).
- [XGBoost parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html), weighted logistic objectives.
- [SHAP TreeExplainer reference](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), shared backgrounds and raw output units.
