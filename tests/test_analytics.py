import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from data import make_demo_data, clean_data, DataQualityError
from analytics import (
    order_level, match_cancellations, cancellation_metrics,
    product_cancellation_hotspots, fit_anomaly_model, risk_engine,
    fit_cancellation_model,
)
from metrics import evaluate_cancellation_model


def test_pipeline_and_matched_cancellation_logic():
    df = make_demo_data(400, seed=42)
    orders = order_level(df)
    assert len(orders) > 100
    match_df, _ = match_cancellations(df)
    metrics = cancellation_metrics(df, match_df)
    assert metrics["matched_original_orders"] > 0
    assert 0 < metrics["matched_cancellation_rate"] < 30
    hotspots = product_cancellation_hotspots(df, match_df, min_orders=5)
    assert "CancellationRate" in hotspots.columns


def test_loader_rejects_collapsed_invoice_column():
    bad = pd.DataFrame({"InvoiceNo": [None, None, None], "StockCode": ["A", "A", "A"], "Quantity": [1, 2, 3], "InvoiceDate": pd.date_range("2020-01-01", periods=3), "UnitPrice": [1, 1, 1]})
    try:
        clean_data(bad)
        assert False, "Expected DataQualityError"
    except DataQualityError:
        pass


def test_anonymous_customers_have_zero_history():
    df = make_demo_data(300, seed=7)
    # Blank a subset of customer IDs in the raw transaction rows.
    identified = df["CustomerID"].notna()
    to_blank = df.loc[identified, "CustomerID"].drop_duplicates().head(10).tolist()
    df.loc[df["CustomerID"].isin(to_blank), "CustomerID"] = pd.NA
    orders = order_level(df)
    anon = orders[orders["IsAnonymous"]]
    assert not anon.empty
    assert (anon["PriorOrders"] == 0).all()
    assert (anon["PriorRevenue"] == 0).all()
    assert (anon["PriorCancellations"] == 0).all()
    assert (anon["PriorCancellationRate"] == 0).all()


def test_customer_cancellation_history_is_prior_only():
    df = make_demo_data(300, seed=9)
    orders = order_level(df)
    identified = orders[~orders["IsAnonymous"]]
    assert (identified["PriorCancellations"] >= 0).all()
    assert (identified["PriorCancellationRate"] >= 0).all()
    assert (identified["PriorCancellationRate"] <= 1).all()


def test_risk_is_repeatable_and_outcome_free():
    df = make_demo_data(300, seed=7)
    orders = order_level(df)
    bundle = fit_anomaly_model(orders)
    scored1 = risk_engine(orders.head(10), bundle)
    scored2 = risk_engine(orders.head(10), bundle)
    pd.testing.assert_series_equal(scored1["RiskScore"], scored2["RiskScore"], check_names=False)
    assert scored1["RiskScore"].between(0, 100).all()


def test_top_decile_metrics_do_not_depend_on_arbitrary_half_threshold():
    y = np.array([1] * 10 + [0] * 90)
    p = np.array([0.9] * 10 + [0.1] * 90)
    m = evaluate_cancellation_model(y, p, baseline_rate=0.10)
    assert m["top10_precision"] == 1.0
    assert m["top10_recall"] == 1.0
    assert m["average_precision"] > 0.9
    assert m["roc_auc"] > 0.9


def test_supervised_cancellation_model_uses_identified_customer_labels():
    df = make_demo_data(600, seed=11)
    orders = order_level(df)
    match_df, _ = match_cancellations(df)
    bundle = fit_cancellation_model(orders, match_df, cancellation_dates=df.loc[df["IsCancellationInvoice"], "InvoiceDate"])
    assert bundle is not None
    _, metrics, labeled, test = bundle
    assert metrics["test_rows"] > 0
    assert "CancelledOutcome" in labeled.columns
    assert "IsAnonymous" in labeled.columns
    assert not labeled["IsAnonymous"].any()
    assert 0 <= metrics["top10_precision"] <= 1
    assert 0 <= metrics["top10_recall"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
