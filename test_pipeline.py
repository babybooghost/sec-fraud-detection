"""Accounting identities, temporal boundaries, weighted costs, and real model tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import (
    ACCOUNTING_COLUMNS,
    BENEISH_COLUMNS,
    FEATURE_COLUMNS,
    FraudConfig,
    FraudPipeline,
)


@pytest.fixture
def panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic annual accounts with firm-level and year-level variation."""
    rng = np.random.default_rng(21)
    rows: list[dict[str, object]] = []
    labels: list[dict[str, int]] = []
    for cik in range(1, 25):
        revenue = float(rng.uniform(100, 300))
        for year in range(2013, 2024):
            fraud = int((cik + year) % 7 == 0)
            revenue *= rng.uniform(0.92, 1.18)
            assets = revenue * rng.uniform(1.2, 1.8)
            rows.append(
                {
                    "cik": cik,
                    "fiscal_year": year,
                    "filing_date": f"{year + 1}-03-01",
                    "revenue": revenue,
                    "receivables": revenue * (0.10 + fraud * 0.18),
                    "cogs": revenue * rng.uniform(0.45, 0.7),
                    "current_assets": assets * 0.3,
                    "ppe_net": assets * rng.uniform(0.3, 0.5),
                    "total_assets": assets,
                    "depreciation": assets * rng.uniform(0.02, 0.06),
                    "sga": revenue * 0.13,
                    "current_liabilities": assets * 0.15,
                    "long_term_debt": assets * 0.2,
                    "income_continuing": revenue * 0.12,
                    "operating_cash_flow": revenue * (0.10 - fraud * 0.12),
                }
            )
            if fraud:
                labels.append({"CIK": cik, "YEARA": year})
    return pd.DataFrame(rows), pd.DataFrame(labels)


@pytest.fixture
def pipeline(panel: tuple[pd.DataFrame, pd.DataFrame]) -> FraudPipeline:
    """Prepare features without fitting a model."""
    pipeline = FraudPipeline(FraudConfig(2019, 2020, n_estimators=30, shap_sample_size=1000))
    pipeline.load_data(*panel)
    pipeline.calculate_beneish_metrics()
    return pipeline


def test_eight_ratios_against_hand_calculation() -> None:
    """Check all formula directions and the cash-flow TATA convention."""
    previous = [100, 10, 60, 40, 50, 150, 10, 20, 20, 30, 12, 10]
    current = [120, 18, 78, 44, 55, 180, 8, 30, 24, 48, 15, 9]
    frame = pd.DataFrame([previous, current], columns=ACCOUNTING_COLUMNS)
    frame["cik"] = 1
    frame["fiscal_year"] = [2018, 2019]
    frame["filing_date"] = ["2019-02-01", "2020-02-01"]
    pipeline = FraudPipeline(FraudConfig(2019, 2020))
    pipeline.load_data(frame, pd.DataFrame(columns=["CIK", "YEARA"]))
    row = pipeline.calculate_beneish_metrics().iloc[1]
    expected = [1.5, 8 / 7, 1.125, 1.2, 21 / 16, 1.25, 1.2, 1 / 30]
    np.testing.assert_allclose(row[list(BENEISH_COLUMNS)].to_numpy(dtype=float), expected)
    assert row.receivables_growth == pytest.approx(0.1)
    assert row.accruals_squared == pytest.approx(1 / 900)


def test_native_bao_join_deduplicates_releases(panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """CIK padding and repeated AAER releases cannot multiply statements."""
    statements, labels = panel
    labels = pd.concat([labels, labels.iloc[:4]], ignore_index=True)
    loaded = FraudPipeline(FraudConfig(2019, 2020)).load_data(statements, labels)
    assert len(loaded) == len(statements)
    assert loaded.cik.str.len().eq(10).all()
    assert loaded.fraud.sum() == len(panel[1])
    assert "fraud" not in statements


def test_gaps_and_cross_firm_lags(panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Never compare different firms or skip missing fiscal years."""
    statements, labels = panel
    statements = statements.loc[~((statements.cik == 1) & (statements.fiscal_year == 2016))]
    pipeline = FraudPipeline(FraudConfig(2019, 2020))
    pipeline.load_data(statements.sample(frac=1, random_state=9), labels)
    features = pipeline.calculate_beneish_metrics()
    first = features.groupby("cik").head(1)
    assert first[list(BENEISH_COLUMNS[:-1])].isna().all().all()
    row = features.loc[(features.cik == "0000000001") & (features.fiscal_year == 2017)].iloc[0]
    assert not row.has_prior_year
    assert pd.isna(row.DSRI)


def test_future_lag_filing_is_not_used(panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """An anomalously late prior filing cannot leak into an earlier decision."""
    statements, labels = panel
    statements.loc[(statements.cik == 1) & (statements.fiscal_year == 2015), "filing_date"] = (
        "2022-01-01"
    )
    pipeline = FraudPipeline(FraudConfig(2019, 2020))
    pipeline.load_data(statements, labels)
    row = (
        pipeline.calculate_beneish_metrics()
        .query("cik == '0000000001' and fiscal_year == 2016")
        .iloc[0]
    )
    assert not row.has_prior_year
    assert pd.isna(row.SGI)


@pytest.mark.parametrize(
    "column", ["revenue", "total_assets", "depreciation", "receivables", "sga"]
)
def test_zero_denominators_stay_missing(
    panel: tuple[pd.DataFrame, pd.DataFrame], column: str
) -> None:
    """Zero inputs never produce infinity or synthetic replacement ratios."""
    statements, labels = panel
    statements[column] = 0.0
    pipeline = FraudPipeline(FraudConfig(2019, 2020))
    pipeline.load_data(statements, labels)
    features = pipeline.calculate_beneish_metrics()[list(FEATURE_COLUMNS)]
    assert not np.isinf(features.to_numpy()).any()
    assert features.isna().any().any()


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "missing_column",
        "infinite",
        "bad_year",
        "missing_cik",
        "bad_date",
        "embedded_label",
    ],
)
def test_bad_statements_rejected(panel: tuple[pd.DataFrame, pd.DataFrame], defect: str) -> None:
    """Reject invalid schemas instead of silently repairing research data."""
    statements, labels = panel
    if defect == "duplicate":
        statements = pd.concat([statements, statements.iloc[:1]])
    elif defect == "missing_column":
        statements = statements.drop(columns="sga")
    elif defect == "infinite":
        statements.loc[0, "sga"] = np.inf
    elif defect == "bad_year":
        statements["fiscal_year"] = statements.fiscal_year.astype(float)
        statements.loc[0, "fiscal_year"] = 2013.5
    elif defect == "missing_cik":
        statements.loc[0, "cik"] = np.nan
    elif defect == "bad_date":
        statements.loc[0, "filing_date"] = None
    else:
        statements["fraud"] = 0
    with pytest.raises(ValueError):
        FraudPipeline(FraudConfig(2019, 2020)).load_data(statements, labels)


def test_invalid_labels_rejected(panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """AAER labels must be explicit binary outcomes."""
    statements, labels = panel
    labels["fraud"] = 2
    with pytest.raises(ValueError, match="binary"):
        FraudPipeline(FraudConfig(2019, 2020)).load_data(statements, labels)


def test_strict_fiscal_and_publication_split(pipeline: FraudPipeline) -> None:
    """Historical fiscal years alone are insufficient when filings arrive late."""
    train, test = pipeline.split_oot()
    assert train.fiscal_year.max() < test.fiscal_year.min()
    assert train.filing_date.max() < pd.Timestamp("2020-01-01", tz="UTC")
    assert test.filing_date.min() >= pd.Timestamp("2020-01-01", tz="UTC")
    assert 2019 not in train.fiscal_year.to_numpy()
    assert set(train.index).isdisjoint(test.index)


def test_cost_weight_and_utility(pipeline: FraudPipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the weighted objective and exact asymmetric accounting."""
    model = pipeline.train_xgboost()
    assert model.get_params()["scale_pos_weight"] == 10.0
    target = pipeline.oot_data.fraud.to_numpy()
    scores = np.where(np.arange(len(target)) % 3 == 0, 0.8, 0.2)
    monkeypatch.setattr(model, "predict_proba", lambda _: np.column_stack([1 - scores, scores]))
    report = pipeline.evaluate_oot()
    fn = int(((scores < 0.5) & (target == 1)).sum())
    fp = int(((scores >= 0.5) & (target == 0)).sum())
    assert report.false_negatives == fn
    assert report.false_positives == fp
    assert report.total_cost == 10 * fn + fp
    assert report.utility == -(10 * fn + fp) / len(target)


def test_future_changes_cannot_change_training(panel: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Future accounting inputs and labels never affect the historical fit."""
    statements, labels = panel
    first = FraudPipeline(FraudConfig(2019, 2020, n_estimators=20))
    first.load_data(statements, labels)
    first.calculate_beneish_metrics()
    first.train_xgboost()
    changed = statements.copy()
    changed.loc[changed.fiscal_year >= 2020, "receivables"] *= 100
    second = FraudPipeline(first.config)
    second.load_data(changed, labels.loc[labels.YEARA < 2020])
    second.calculate_beneish_metrics()
    second.train_xgboost()
    pd.testing.assert_frame_equal(first.train_data, second.train_data)
    np.testing.assert_array_equal(
        first.model.predict_proba(first.train_data[list(FEATURE_COLUMNS)]),
        second.model.predict_proba(second.train_data[list(FEATURE_COLUMNS)]),
    )


def test_real_shap_drift_decomposes_margin(
    pipeline: FraudPipeline, capsys: pytest.CaptureFixture[str]
) -> None:
    """A common background makes signed SHAP shifts sum to prediction drift."""
    pipeline.train_xgboost()
    audit = pipeline.run_shap_audit()
    train_margin = pipeline.model.predict(
        pipeline.train_data[list(FEATURE_COLUMNS)], output_margin=True
    )
    test_margin = pipeline.model.predict(
        pipeline.oot_data[list(FEATURE_COLUMNS)], output_margin=True
    )
    assert audit.mean_shap_shift.sum() == pytest.approx(
        float(test_margin.mean() - train_margin.mean()), abs=1e-5
    )
    assert len(audit) == len(FEATURE_COLUMNS)
    assert audit.absolute_mean_shift.is_monotonic_decreasing
    assert np.isfinite(audit.select_dtypes("number")).all().all()
    assert len(capsys.readouterr().out.strip().splitlines()) == 6


def test_single_class_oot_auc_is_none(pipeline: FraudPipeline) -> None:
    """Undefined AUC should not warn or invent a score."""
    pipeline.train_xgboost()
    pipeline.oot_data["fraud"] = 0
    assert pipeline.evaluate_oot().roc_auc is None


def test_reload_invalidates_model(
    pipeline: FraudPipeline, panel: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Prevent stale predictions after changing the input dataset."""
    pipeline.train_xgboost()
    pipeline.load_data(*panel)
    with pytest.raises(RuntimeError):
        pipeline.evaluate_oot()


@pytest.mark.parametrize(
    "method", ["calculate_beneish_metrics", "train_xgboost", "evaluate_oot", "run_shap_audit"]
)
def test_invalid_call_order(method: str) -> None:
    """Methods fail with lifecycle errors before data exists."""
    with pytest.raises(RuntimeError):
        getattr(FraudPipeline(FraudConfig(2019, 2020)), method)()


@pytest.mark.parametrize(
    "options",
    [
        {"train_end_year": 2020, "test_start_year": 2020},
        {"train_end_year": 2019, "test_start_year": 2020, "false_negative_cost": 0},
        {"train_end_year": 2019, "test_start_year": 2020, "decision_threshold": 1},
        {"train_end_year": 2019, "test_start_year": 2020, "test_end_year": 2018},
    ],
)
def test_invalid_configuration(options: dict[str, int]) -> None:
    """Configuration cannot create overlapping or nonsensical experiments."""
    with pytest.raises(ValueError):
        FraudConfig(**options)


def test_cli_end_to_end(panel: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    """Run the real CSV, model, JSON report, and SHAP command interface."""
    statements, labels = panel
    statements.to_csv(tmp_path / "financials.csv", index=False)
    labels.to_csv(tmp_path / "aaer.csv", index=False)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("pipeline.py")),
            "--financials",
            str(tmp_path / "financials.csv"),
            "--aaer",
            str(tmp_path / "aaer.csv"),
            "--train-end-year",
            "2019",
            "--test-start-year",
            "2020",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout.splitlines()[0])
    assert report["observations"] > 0
    assert "shap_drift_audit" in result.stderr
