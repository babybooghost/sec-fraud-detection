"""Temporal fraud research on normalized, point-in-time SEC annual statements.

Run ``python pipeline.py --help`` for the CSV interface. AAER labels are
retrospective outcomes, not proof that fraud was observable when a filing arrived.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)
FrameSource = pd.DataFrame | str | Path
ACCOUNTING_COLUMNS = (
    "revenue",
    "receivables",
    "cogs",
    "current_assets",
    "ppe_net",
    "total_assets",
    "depreciation",
    "sga",
    "current_liabilities",
    "long_term_debt",
    "income_continuing",
    "operating_cash_flow",
)
BENEISH_COLUMNS = ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA")
INTERACTION_COLUMNS = (
    "receivables_growth",
    "margin_growth",
    "asset_quality_accruals",
    "leverage_accruals",
    "accruals_squared",
    "growth_squared",
)
FEATURE_COLUMNS = BENEISH_COLUMNS + INTERACTION_COLUMNS


class JsonFormatter(logging.Formatter):
    """Serialize pipeline events as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Include structured event details and any exception traceback."""
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "details": getattr(record, "details", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, allow_nan=False)


@dataclass(frozen=True)
class FraudConfig:
    """Fix the split and decision rule before examining forward outcomes.

    Training filings must be available by December 31 of ``train_end_year``.
    ``test_end_year`` bounds fiscal years, not filing publication dates.
    """

    train_end_year: int
    test_start_year: int
    test_end_year: int | None = None
    false_negative_cost: float = 10.0
    false_positive_cost: float = 1.0
    decision_threshold: float = 0.5
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 42
    shap_sample_size: int = 1000
    shap_background_size: int = 100

    def __post_init__(self) -> None:
        """Reject overlapping windows and invalid model settings."""
        if not 1900 <= self.train_end_year < self.test_start_year <= 2200:
            raise ValueError("Expected 1900 <= train_end_year < test_start_year <= 2200")
        if self.test_end_year is not None and self.test_end_year < self.test_start_year:
            raise ValueError("test_end_year precedes test_start_year")
        if not all(
            np.isfinite(c) and c > 0
            for c in (
                self.false_negative_cost,
                self.false_positive_cost,
            )
        ):
            raise ValueError("Misclassification costs must be finite and positive")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("decision_threshold must be between zero and one")
        if (
            min(self.n_estimators, self.max_depth, self.shap_sample_size, self.shap_background_size)
            < 1
        ):
            raise ValueError("Tree counts, depth, and SHAP sample sizes must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")


@dataclass(frozen=True)
class FraudEvaluation:
    """Forward-set metrics; utility is negative cost per observation."""

    observations: int
    precision: float
    recall: float
    roc_auc: float | None
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    total_cost: float
    utility: float


class FraudPipeline:
    """Engineer annual ratios, fit historical data, and audit forward drift.

    Input statements must use consistent units and accounting definitions, with
    one original filing per CIK and fiscal year. Restatement resolution and XBRL
    taxonomy mapping belong upstream. Identifiers and labels never enter X.
    """

    def __init__(self, config: FraudConfig) -> None:
        """Initialize explicit configuration and unfitted state."""
        self.config = config
        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.oot_data: pd.DataFrame | None = None
        self.model: XGBClassifier | None = None
        self.shap_audit: pd.DataFrame | None = None

    @staticmethod
    def _read(source: FrameSource) -> pd.DataFrame:
        """Read CSV or copy a caller-owned frame without mutating it."""
        return source.copy(deep=True) if isinstance(source, pd.DataFrame) else pd.read_csv(source)

    @staticmethod
    def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric SEC CIKs and integer fiscal years."""
        for column in ("cik", "fiscal_year"):
            if column not in frame:
                raise ValueError(f"Missing required column: {column}")
            values = pd.to_numeric(frame[column], errors="raise")
            if values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any():
                raise ValueError(f"{column} must contain finite integers")
            frame[column] = values.astype("int64")
        if not frame.cik.between(1, 9_999_999_999).all():
            raise ValueError("CIK must be a positive SEC identifier of at most ten digits")
        if not frame.fiscal_year.between(1900, 2200).all():
            raise ValueError("Fiscal years must lie between 1900 and 2200")
        frame["cik"] = frame.cik.astype(str).str.zfill(10)
        return frame

    def load_data(self, financials: FrameSource, aaer: FrameSource) -> pd.DataFrame:
        """Join annual statements to Bao AAER positive firm-year observations.

        Bao's native ``CIK,YEARA`` columns are accepted. An optional ``fraud``
        column must be binary; without it, every AAER row is a positive label.
        Repeated enforcement releases collapse to a single firm-year maximum.
        Unmatched statements receive zero, meaning no observed AAER event.
        """
        self.data = self.features = self.train_data = self.oot_data = None
        self.model = None
        self.shap_audit = None
        statements = self._normalize_keys(self._read(financials))
        required = set(ACCOUNTING_COLUMNS) | {"filing_date"}
        missing = required - set(statements)
        if missing:
            raise ValueError(f"Missing statement columns: {sorted(missing)}")
        if statements.empty:
            raise ValueError("Statement data is empty")
        if "fraud" in statements:
            raise ValueError("Supply fraud labels through the AAER input only")
        if statements.duplicated(["cik", "fiscal_year"]).any():
            raise ValueError("Duplicate firm-year statements; resolve filing vintages upstream")
        statements["filing_date"] = pd.to_datetime(statements.filing_date, utc=True, errors="raise")
        if statements.filing_date.isna().any():
            raise ValueError("filing_date cannot be missing")
        if (statements.filing_date.dt.year < statements.fiscal_year).any():
            raise ValueError("A filing cannot precede its fiscal year")
        for column in ACCOUNTING_COLUMNS:
            statements[column] = pd.to_numeric(statements[column], errors="raise").astype(float)
        if np.isinf(statements[list(ACCOUNTING_COLUMNS)].to_numpy()).any():
            raise ValueError("Accounting inputs cannot contain infinity")
        labels = self._read(aaer).rename(columns={"CIK": "cik", "YEARA": "fiscal_year"})
        labels = self._normalize_keys(labels)
        if "fraud" not in labels:
            labels["fraud"] = 1
        if not labels.fraud.isin([0, 1]).all():
            raise ValueError("AAER fraud labels must be binary and nonmissing")
        labels = labels.groupby(["cik", "fiscal_year"], as_index=False).fraud.max()
        statements = statements.merge(
            labels, on=["cik", "fiscal_year"], how="left", validate="one_to_one"
        )
        statements["fraud"] = statements.fraud.fillna(0).astype("int8")
        self.data = statements.sort_values(["cik", "fiscal_year"]).reset_index(drop=True)
        LOGGER.info(
            "data_loaded",
            extra={"details": {"rows": len(self.data), "fraud_rows": int(self.data.fraud.sum())}},
        )
        return self.data.copy()

    @staticmethod
    def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Return NaN for missing, near-zero, or overflowing ratios."""
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = numerator / denominator.where(denominator.abs() > 1e-12)
        return result.replace([np.inf, -np.inf], np.nan)

    def calculate_beneish_metrics(self) -> pd.DataFrame:
        """Calculate eight ratios with consecutive, already-published annual lags.

        TATA uses (income from continuing operations minus operating cash flow)
        divided by assets. Undefined ratios remain missing for XGBoost's native
        missing-value routing. No imputation or clipping sees future data.
        """
        if self.data is None:
            raise RuntimeError("Call load_data before calculating features")
        self.model = None
        self.train_data = self.oot_data = None
        self.shap_audit = None
        frame = self.data.copy()
        grouped = frame.groupby("cik", sort=False)
        lag = grouped[list(ACCOUNTING_COLUMNS)].shift(1)
        valid_lag = frame.fiscal_year.sub(grouped.fiscal_year.shift(1)).eq(
            1
        ) & grouped.filing_date.shift(1).le(frame.filing_date)
        lag.loc[~valid_lag, :] = np.nan
        div = self._divide
        frame["DSRI"] = div(
            div(frame.receivables, frame.revenue), div(lag.receivables, lag.revenue)
        )
        frame["GMI"] = div(
            div(lag.revenue - lag.cogs, lag.revenue), div(frame.revenue - frame.cogs, frame.revenue)
        )
        frame["AQI"] = div(
            1 - div(frame.current_assets + frame.ppe_net, frame.total_assets),
            1 - div(lag.current_assets + lag.ppe_net, lag.total_assets),
        )
        frame["SGI"] = div(frame.revenue, lag.revenue)
        frame["DEPI"] = div(
            div(lag.depreciation, lag.depreciation + lag.ppe_net),
            div(frame.depreciation, frame.depreciation + frame.ppe_net),
        )
        frame["SGAI"] = div(div(frame.sga, frame.revenue), div(lag.sga, lag.revenue))
        frame["LVGI"] = div(
            div(frame.current_liabilities + frame.long_term_debt, frame.total_assets),
            div(lag.current_liabilities + lag.long_term_debt, lag.total_assets),
        )
        frame["TATA"] = div(frame.income_continuing - frame.operating_cash_flow, frame.total_assets)
        with np.errstate(over="ignore", invalid="ignore"):
            frame["receivables_growth"] = (frame.DSRI - 1) * (frame.SGI - 1)
            frame["margin_growth"] = (frame.GMI - 1) * (frame.SGI - 1)
            frame["asset_quality_accruals"] = (frame.AQI - 1) * frame.TATA
            frame["leverage_accruals"] = (frame.LVGI - 1) * frame.TATA
            frame["accruals_squared"] = frame.TATA.pow(2)
            frame["growth_squared"] = (frame.SGI - 1).pow(2)
        frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].replace(
            [np.inf, -np.inf], np.nan
        )
        # float32 is the model's input representation. Unrepresentable values stay missing.
        frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].where(
            frame[list(FEATURE_COLUMNS)].abs() <= np.finfo(np.float32).max
        )
        frame["has_prior_year"] = valid_lag
        self.features = frame
        return frame.copy()

    def split_oot(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Enforce fiscal-year and publication-time separation without shuffling."""
        if self.features is None:
            raise RuntimeError("Call calculate_beneish_metrics before splitting")
        cutoff = pd.Timestamp(f"{self.config.train_end_year + 1}-01-01", tz="UTC")
        frame = self.features
        eligible = frame.has_prior_year & frame[list(FEATURE_COLUMNS)].notna().any(axis=1)
        train_mask = (
            eligible
            & frame.fiscal_year.le(self.config.train_end_year)
            & frame.filing_date.lt(cutoff)
        )
        test_mask = (
            eligible
            & frame.fiscal_year.ge(self.config.test_start_year)
            & frame.filing_date.ge(cutoff)
        )
        if self.config.test_end_year is not None:
            test_mask &= frame.fiscal_year.le(self.config.test_end_year)
        train = frame.loc[train_mask].sort_values(["filing_date", "cik"]).copy()
        test = frame.loc[test_mask].sort_values(["filing_date", "cik"]).copy()
        if train.empty or test.empty:
            raise ValueError("Temporal split must produce nonempty train and OOT sets")
        if train.fraud.nunique() != 2:
            raise ValueError("Training requires both fraud and non-fraud observations")
        self.train_data, self.oot_data = train, test
        LOGGER.info(
            "temporal_split",
            extra={
                "details": {
                    "train_rows": len(train),
                    "oot_rows": len(test),
                    "excluded_rows": len(frame) - len(train) - len(test),
                    "cutoff": cutoff,
                }
            },
        )
        return train.copy(), test.copy()

    def train_xgboost(self) -> XGBClassifier:
        """Fit weighted logistic loss using a 10:1 positive-to-negative cost.

        This differentiable surrogate penalizes missed fraud more heavily.
        A threshold of 0.5 applies to the weighted model score, which is not a
        calibrated fraud probability. Do not apply a second 1/11 threshold.
        """
        self.model = None
        self.shap_audit = None
        train, _ = self.split_oot()
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=1.0,
            colsample_bytree=1.0,
            scale_pos_weight=self.config.false_negative_cost / self.config.false_positive_cost,
            random_state=self.config.random_state,
            n_jobs=1,
        )
        model.fit(train[list(FEATURE_COLUMNS)], train.fraud)
        self.model = model
        LOGGER.info(
            "model_trained",
            extra={"details": {"rows": len(train), "features": len(FEATURE_COLUMNS)}},
        )
        return model

    def evaluate_oot(self) -> FraudEvaluation:
        """Evaluate untouched forward data with utility = -(10 FN + FP) / N."""
        if self.model is None or self.oot_data is None:
            raise RuntimeError("Call train_xgboost before evaluation")
        target = self.oot_data.fraud
        scores = self.model.predict_proba(self.oot_data[list(FEATURE_COLUMNS)])[:, 1]
        predictions = scores >= self.config.decision_threshold
        tn, fp, fn, tp = (
            int(x) for x in confusion_matrix(target, predictions, labels=[0, 1]).ravel()
        )
        cost = self.config.false_negative_cost * fn + self.config.false_positive_cost * fp
        report = FraudEvaluation(
            observations=len(target),
            precision=float(precision_score(target, predictions, zero_division=0)),
            recall=float(recall_score(target, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(target, scores)) if target.nunique() == 2 else None,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            true_positives=tp,
            total_cost=cost,
            utility=-cost / len(target),
        )
        LOGGER.info("oot_evaluation", extra={"details": asdict(report)})
        return report

    @staticmethod
    def _psi(reference: np.ndarray, observed: np.ndarray) -> float:
        """Compute descriptive SHAP distribution PSI using training-only bins."""
        interior = np.unique(np.quantile(reference, np.linspace(0.1, 0.9, 9)))
        edges = np.concatenate(([-np.inf], interior, [np.inf]))
        left = np.histogram(reference, bins=edges)[0].astype(float) + 0.5
        right = np.histogram(observed, bins=edges)[0].astype(float) + 0.5
        left /= left.sum()
        right /= right.sum()
        return float(np.sum((right - left) * np.log(right / left)))

    def run_shap_audit(self, *, print_top: bool = True) -> pd.DataFrame:
        """Rank signed mean SHAP shifts and report magnitude/distribution drift.

        Both sets use one training-only background and raw log-odds units.
        Signed mean shifts decompose the change in mean model margin. PSI and
        magnitude shifts are descriptive diagnostics, not causal evidence or
        significance tests. Sampling is deterministic and bounded.
        """
        if self.model is None or self.train_data is None or self.oot_data is None:
            raise RuntimeError("Call train_xgboost before SHAP audit")
        seed = self.config.random_state
        train = self.train_data[list(FEATURE_COLUMNS)].sample(
            min(len(self.train_data), self.config.shap_sample_size), random_state=seed
        )
        test = self.oot_data[list(FEATURE_COLUMNS)].sample(
            min(len(self.oot_data), self.config.shap_sample_size), random_state=seed
        )
        background = self.train_data[list(FEATURE_COLUMNS)].sample(
            min(len(self.train_data), self.config.shap_background_size), random_state=seed
        )
        explainer = shap.TreeExplainer(
            self.model, data=background, feature_perturbation="interventional", model_output="raw"
        )
        reference = np.asarray(explainer(train).values, dtype=float)
        observed = np.asarray(explainer(test).values, dtype=float)
        audit = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "train_mean_abs_shap": np.abs(reference).mean(axis=0),
                "oot_mean_abs_shap": np.abs(observed).mean(axis=0),
                "mean_shap_shift": observed.mean(axis=0) - reference.mean(axis=0),
                "mean_abs_shap_shift": np.abs(observed).mean(axis=0)
                - np.abs(reference).mean(axis=0),
                "shap_psi": [
                    self._psi(reference[:, i], observed[:, i]) for i in range(len(FEATURE_COLUMNS))
                ],
            }
        )
        audit["absolute_mean_shift"] = audit.mean_shap_shift.abs()
        self.shap_audit = audit.sort_values(
            "absolute_mean_shift", ascending=False, kind="stable"
        ).reset_index(drop=True)
        if print_top:
            print(self.shap_audit.head(5).to_string(index=False))
        LOGGER.info(
            "shap_drift_audit",
            extra={
                "details": {
                    "train_sample": len(train),
                    "oot_sample": len(test),
                    "mean_margin_shift": float(audit.mean_shap_shift.sum()),
                    "top_drivers": self.shap_audit.head(5).feature.tolist(),
                }
            },
        )
        return self.shap_audit.copy()


def main(argv: list[str] | None = None) -> int:
    """Execute the local CSV workflow and print machine-readable metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financials", type=Path, required=True)
    parser.add_argument("--aaer", type=Path, required=True)
    parser.add_argument("--train-end-year", type=int, required=True)
    parser.add_argument("--test-start-year", type=int, required=True)
    parser.add_argument("--test-end-year", type=int)
    args = parser.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    pipeline = FraudPipeline(
        FraudConfig(args.train_end_year, args.test_start_year, args.test_end_year)
    )
    pipeline.load_data(args.financials, args.aaer)
    pipeline.calculate_beneish_metrics()
    pipeline.train_xgboost()
    print(json.dumps(asdict(pipeline.evaluate_oot()), allow_nan=False))
    pipeline.run_shap_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
