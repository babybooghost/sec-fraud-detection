"""Validate the published synthetic experiment against its held-out predictions."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).parent
DATA = json.loads((ROOT / "docs" / "data.json").read_text())


def test_demo_provenance_and_alignment() -> None:
    """Every published prediction must have a synthetic record and binary target."""
    assert DATA["synthetic"] is True
    assert "synthetic" in DATA["provenance"].lower()
    assert len(DATA["truth"]) == len(DATA["records"]) == DATA["test_count"]
    assert {0, 1} == set(DATA["truth"])
    assert [row["target"] for row in DATA["records"]] == DATA["truth"]
    assert DATA["train_count"] > 0


@pytest.mark.parametrize("key", DATA["models"])
def test_saved_metrics_match_predictions(key: str) -> None:
    """Independently reconstruct ranking metrics, including tied ROC points."""
    model = DATA["models"][key]
    scores = np.asarray(model["scores"])
    assert scores.shape == (DATA["test_count"],)
    assert np.isfinite(scores).all()
    assert model["roc_auc"] == pytest.approx(roc_auc_score(DATA["truth"], scores))
    assert model["pr_auc"] == pytest.approx(average_precision_score(DATA["truth"], scores))
    fpr, tpr, _ = roc_curve(DATA["truth"], scores)
    np.testing.assert_allclose(model["roc"], np.column_stack([fpr, tpr]))


def test_representation_metadata() -> None:
    """Representations must agree with exported dimensions and partition sizes."""
    if DATA["kind"] == "reddit":
        assert 0.95 <= sum(DATA["variance"]) <= 1.000001
        assert len(DATA["timeline"]) == len(DATA["projection"])
        for split, count in [
            ("train", DATA["train_count"]),
            ("holdout", DATA["test_count"]),
            ("purged", DATA["purged_count"]),
        ]:
            assert sum(row["split"] == split for row in DATA["projection"]) == count
    else:
        assert {row["feature"] for row in DATA["attribution"]} == set(DATA["features"])
        assert len(DATA["features"]) == 14
        for row in DATA["attribution"]:
            assert np.isfinite(row["oot_mean_abs_shap"])
            assert row["oot_mean_abs_shap"] >= 0
