from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from metrics import evaluate_cancellation_model

ANOMALY_FEATURES = [
    "OrderValue", "TotalQuantity", "UniqueProducts", "PriorOrders", "PriorRevenue", "DaysSincePriorOrder",
    "PriorCancellations", "PriorCancellationRate",
]
CANCEL_FEATURES_NUMERIC = [
    "OrderValue", "TotalQuantity", "UniqueProducts", "PriorOrders", "PriorRevenue",
    "DaysSincePriorOrder", "PriorCancellations", "PriorCancellationRate", "Hour", "Weekday",
]
CANCEL_FEATURES_CATEGORICAL = ["Country"]


def sales_lines(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(~df["IsCancellationInvoice"]) & (df["Quantity"] > 0)].copy()


def order_level(df: pd.DataFrame) -> pd.DataFrame:
    """Build order-level records from genuine positive sales invoices only."""
    x = sales_lines(df)
    grp = x.groupby("InvoiceNo", dropna=False)
    orders = grp.agg(
        OrderValue=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        InvoiceDate=("InvoiceDate", "first"),
    ).reset_index()
    orders["IsAnonymous"] = orders["CustomerID"].isna()
    orders["Hour"] = orders["InvoiceDate"].dt.hour
    orders["Weekday"] = orders["InvoiceDate"].dt.weekday
    orders["Month"] = orders["InvoiceDate"].dt.to_period("M").astype(str)
    return add_customer_history(orders, df)


def add_customer_history(orders: pd.DataFrame, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add leakage-safe customer history. Anonymous customers receive zero history."""
    x = orders.copy().sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
    x["IsAnonymous"] = x["CustomerID"].isna()
    identified = ~x["IsAnonymous"]

    x["PriorOrders"] = 0
    x["PriorRevenue"] = 0.0
    x["DaysSincePriorOrder"] = 3650.0
    x.loc[identified, "PriorOrders"] = (
        x.loc[identified].groupby("CustomerID", sort=False).cumcount()
    )
    prior_rev = x.loc[identified].groupby("CustomerID", sort=False)["OrderValue"].cumsum() - x.loc[identified, "OrderValue"]
    x.loc[identified, "PriorRevenue"] = prior_rev.to_numpy()
    prev = x.loc[identified].groupby("CustomerID", sort=False)["InvoiceDate"].shift(1)
    gaps = (x.loc[identified, "InvoiceDate"].reset_index(drop=True) - prev.reset_index(drop=True)).dt.total_seconds() / 86400.0
    x.loc[identified, "DaysSincePriorOrder"] = gaps.fillna(3650).clip(lower=0).to_numpy()

    x["PriorCancellations"] = 0
    x["PriorCancellationRate"] = 0.0
    if df is not None:
        cancels = df.loc[df["IsCancellationInvoice"] & df["CustomerID"].notna(), ["CustomerID", "InvoiceDate", "InvoiceNo"]].drop_duplicates()
        if not cancels.empty:
            cancels = cancels.sort_values(["InvoiceDate", "CustomerID", "InvoiceNo"])
            cancels["PriorCancellationCountAtEvent"] = cancels.groupby("CustomerID").cumcount() + 1
            history = x.loc[identified, ["CustomerID", "InvoiceDate"]].copy()
            history["_row_index"] = history.index
            history = history.sort_values(["InvoiceDate", "CustomerID"])
            prior = pd.merge_asof(
                history,
                cancels[["CustomerID", "InvoiceDate", "PriorCancellationCountAtEvent"]].sort_values(["InvoiceDate", "CustomerID"]),
                on="InvoiceDate", by="CustomerID", direction="backward", allow_exact_matches=False,
            )
            prior = prior.set_index("_row_index")["PriorCancellationCountAtEvent"].fillna(0)
            x.loc[prior.index, "PriorCancellations"] = prior.to_numpy()
            x.loc[identified, "PriorCancellationRate"] = (
                x.loc[identified, "PriorCancellations"].to_numpy() /
                np.maximum(x.loc[identified, "PriorOrders"].to_numpy(), 1)
            )

    # Anonymous orders must never inherit another anonymous person's history.
    x.loc[~identified, ["PriorOrders", "PriorRevenue", "PriorCancellations", "PriorCancellationRate"]] = 0
    x.loc[~identified, "DaysSincePriorOrder"] = 3650.0
    return x.sort_index().reset_index(drop=True)


def match_cancellations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match C-invoice lines to likely earlier positive sales lines.

    Matching requires an identified customer, same stock code, earlier date and
    exact absolute quantity. Unmatched/anonymous/partial cases remain unmatched,
    so the matched cancellation rate is explicitly a lower bound.
    """
    sales = sales_lines(df).copy()
    cancels = df.loc[df["IsCancellationInvoice"] & (df["Quantity"] < 0)].copy()
    eligible = cancels.dropna(subset=["CustomerID"]).copy()
    if eligible.empty or sales.empty:
        return pd.DataFrame(columns=["CancellationInvoice", "OriginalInvoice", "MatchedLines", "MatchedQuantity"]), pd.DataFrame()

    sales = sales.dropna(subset=["CustomerID"])
    sales["QtyAbs"] = sales["Quantity"].abs()
    eligible["QtyAbs"] = eligible["Quantity"].abs()
    sales = sales.sort_values("InvoiceDate")
    eligible = eligible.sort_values("InvoiceDate")

    matched = pd.merge_asof(
        eligible[["InvoiceNo", "CustomerID", "StockCode", "QtyAbs", "InvoiceDate"]].rename(columns={"InvoiceNo": "CancellationInvoice"}),
        sales[["InvoiceNo", "CustomerID", "StockCode", "QtyAbs", "InvoiceDate", "Quantity"]].rename(columns={"InvoiceNo": "OriginalInvoice", "InvoiceDate": "OriginalDate"}).sort_values("OriginalDate"),
        left_on="InvoiceDate", right_on="OriginalDate", by=["CustomerID", "StockCode", "QtyAbs"],
        direction="backward", allow_exact_matches=False,
    ).dropna(subset=["OriginalInvoice"])
    if matched.empty:
        return pd.DataFrame(columns=["CancellationInvoice", "OriginalInvoice", "MatchedLines", "MatchedQuantity"]), matched

    summary = (
        matched.groupby(["CancellationInvoice", "OriginalInvoice"], as_index=False)
        .agg(MatchedLines=("StockCode", "size"), MatchedQuantity=("QtyAbs", "sum"), OriginalDate=("OriginalDate", "max"))
        .sort_values(["CancellationInvoice", "MatchedLines", "MatchedQuantity", "OriginalDate"], ascending=[True, False, False, False])
    )
    best = summary.drop_duplicates("CancellationInvoice", keep="first")
    return best[["CancellationInvoice", "OriginalInvoice", "MatchedLines", "MatchedQuantity"]], matched


def cancellation_metrics(df: pd.DataFrame, match_df: pd.DataFrame | None = None) -> dict:
    if match_df is None:
        match_df, _ = match_cancellations(df)
    cancel_invoices = set(df.loc[df["IsCancellationInvoice"], "InvoiceNo"].unique())
    eligible_cancel_invoices = set(df.loc[df["IsCancellationInvoice"] & df["CustomerID"].notna(), "InvoiceNo"].unique())
    positive_orders = set(sales_lines(df)["InvoiceNo"].unique())
    matched_originals = set(match_df["OriginalInvoice"].dropna().unique())
    fully_matched = 0
    if not match_df.empty:
        cancel_line_counts = df.loc[df["IsCancellationInvoice"]].groupby("InvoiceNo").size()
        fully = match_df.merge(cancel_line_counts.rename("TotalCancelLines"), left_on="CancellationInvoice", right_index=True, how="left")
        fully_matched = int((fully["MatchedLines"] >= fully["TotalCancelLines"]).sum())
    return {
        "cancellation_invoices": len(cancel_invoices),
        "eligible_cancellation_invoices": len(eligible_cancel_invoices),
        "matched_cancellation_invoices": int(match_df["CancellationInvoice"].nunique()),
        "matched_original_orders": len(matched_originals),
        "positive_orders": len(positive_orders),
        "matched_cancellation_rate": (len(matched_originals) / len(positive_orders) * 100) if positive_orders else 0.0,
        "raw_cancellation_invoice_rate": (len(cancel_invoices) / max(1, len(positive_orders)) * 100),
        "cancellation_invoice_match_rate": (match_df["CancellationInvoice"].nunique() / max(1, len(eligible_cancel_invoices)) * 100),
        "fully_matched_cancellation_invoices": fully_matched,
    }


def product_cancellation_hotspots(df: pd.DataFrame, match_df: pd.DataFrame, min_orders: int = 20) -> pd.DataFrame:
    sales = sales_lines(df)
    products = sales.loc[sales["IsProduct"]].copy()
    if products.empty:
        return pd.DataFrame(columns=["Product", "Orders", "CancelledOrders", "CancellationRate"])
    cancelled_originals = set(match_df["OriginalInvoice"].dropna())
    product_orders = products.groupby("StockCode")["InvoiceNo"].nunique().rename("Orders")
    product_cancelled = products.loc[products["InvoiceNo"].isin(cancelled_originals)].groupby("StockCode")["InvoiceNo"].nunique().rename("CancelledOrders")
    desc = products.groupby("StockCode")["Description"].first().rename("Product")
    out = pd.concat([desc, product_orders, product_cancelled], axis=1).fillna({"CancelledOrders": 0}).reset_index()
    out["CancellationRate"] = out["CancelledOrders"] / out["Orders"] * 100
    return out.loc[out["Orders"] >= min_orders].sort_values(["CancellationRate", "Orders"], ascending=[False, False]).reset_index(drop=True)


def customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    orders = order_level(df)
    valid = orders.dropna(subset=["CustomerID"]).copy()
    m = valid.groupby("CustomerID").agg(
        Orders=("InvoiceNo", "nunique"), Revenue=("OrderValue", "sum"), AvgOrderValue=("OrderValue", "mean"), LastOrder=("InvoiceDate", "max"),
    ).reset_index()
    m["Segment"] = np.select([m["Orders"] >= 10, m["Orders"] >= 3], ["Frequent", "Repeat"], default="Occasional")
    return m


@dataclass
class AnomalyBundle:
    model: IsolationForest
    feature_names: list[str]
    raw_min: float
    raw_max: float
    reference_values: dict[str, np.ndarray]


def _model_frame(orders: pd.DataFrame) -> pd.DataFrame:
    x = orders[ANOMALY_FEATURES].copy()
    return x.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)


def fit_anomaly_model(orders: pd.DataFrame) -> AnomalyBundle:
    model_data = _model_frame(orders)
    if len(model_data) < 20:
        raise ValueError("At least 20 sales orders are required for anomaly detection")
    contamination = min(0.08, max(0.02, 20 / len(model_data)))
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(model_data)
    raw = -model.score_samples(model_data)
    refs = {col: np.sort(model_data[col].to_numpy()) for col in ANOMALY_FEATURES}
    return AnomalyBundle(model, ANOMALY_FEATURES, float(raw.min()), float(raw.max()), refs)


def _percentile_against_reference(values: pd.Series, ref: np.ndarray) -> np.ndarray:
    if len(ref) <= 1:
        return np.zeros(len(values))
    ranks = np.searchsorted(ref, values.to_numpy(), side="right")
    return ranks / len(ref) * 100


def score_anomalies(orders: pd.DataFrame, bundle: AnomalyBundle) -> pd.DataFrame:
    x = orders.copy()
    model_data = _model_frame(x)
    pred = bundle.model.predict(model_data)
    raw = -bundle.model.score_samples(model_data)
    x["AnomalyScore"] = ((raw - bundle.raw_min) / (bundle.raw_max - bundle.raw_min + 1e-9) * 100).clip(0, 100).round(1)
    x["IsAnomaly"] = pred == -1
    for col in ANOMALY_FEATURES:
        x[f"{col}Percentile"] = _percentile_against_reference(model_data[col], bundle.reference_values[col])
    return x


def add_anomaly_scores(orders: pd.DataFrame) -> pd.DataFrame:
    return score_anomalies(orders, fit_anomaly_model(orders))


def risk_engine(orders: pd.DataFrame, bundle: AnomalyBundle | None = None) -> pd.DataFrame:
    x = orders.copy()
    if bundle is None:
        bundle = fit_anomaly_model(x)
    x = score_anomalies(x, bundle)
    score = (
        0.50 * x["AnomalyScore"] +
        0.20 * x["OrderValuePercentile"] +
        0.12 * x["TotalQuantityPercentile"] +
        0.08 * x["UniqueProductsPercentile"] +
        0.05 * x["PriorCancellationRate"].clip(0, 1) * 100 +
        0.05 * x["PriorCancellations"].clip(0, 5) / 5 * 100
    )
    x["RiskScore"] = score.clip(0, 100).round(1)
    x["RiskLevel"] = pd.cut(x["RiskScore"], bins=[-1, 30, 70, 101], labels=["Low", "Medium", "High"])
    x["Priority"] = x["RiskLevel"].map({"High": "P1 - Review", "Medium": "P2 - Normal", "Low": "P3 - Standard"}).fillna("P3 - Standard")
    return x


def build_cancellation_labels(orders: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    x = orders.copy()
    cancelled = set(match_df["OriginalInvoice"].dropna())
    x["CancelledOutcome"] = x["InvoiceNo"].isin(cancelled).astype(int)
    # Anonymous customers cannot be reliably labelled through customer-linked matching.
    return x.loc[~x["IsAnonymous"]].copy()


def fit_cancellation_model(orders: pd.DataFrame, match_df: pd.DataFrame, cancellation_dates: pd.Series | None = None, censor_days: int = 28):
    """Train/evaluate only identified customers and exclude recent orders whose cancellations may be censored."""
    labeled = build_cancellation_labels(orders, match_df).sort_values("InvoiceDate").reset_index(drop=True)
    if labeled.empty or labeled["CancelledOutcome"].nunique() < 2 or labeled["CancelledOutcome"].sum() < 10:
        return None

    latest_observed = pd.to_datetime(orders["InvoiceDate"]).max()
    if cancellation_dates is not None and len(cancellation_dates):
        latest_cancellation_event = pd.to_datetime(cancellation_dates).max()
    else:
        latest_cancellation_event = latest_observed
    cutoff = latest_cancellation_event - pd.Timedelta(days=censor_days)
    labeled = labeled.loc[labeled["InvoiceDate"] <= cutoff].reset_index(drop=True)
    if len(labeled) < 50 or labeled["CancelledOutcome"].sum() < 10:
        return None

    split = int(len(labeled) * 0.80)
    train, test = labeled.iloc[:split].copy(), labeled.iloc[split:].copy()
    if test["CancelledOutcome"].nunique() < 2:
        return None

    numeric = CANCEL_FEATURES_NUMERIC
    categorical = CANCEL_FEATURES_CATEGORICAL
    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("prep", pre),
        ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    model.fit(train[numeric + categorical], train["CancelledOutcome"])
    prob = model.predict_proba(test[numeric + categorical])[:, 1]
    metrics = evaluate_cancellation_model(test["CancelledOutcome"].to_numpy(), prob, baseline_rate=float(train["CancelledOutcome"].mean()))
    metrics.update({
        "train_rows": len(train), "test_rows": len(test), "positive_train": int(train["CancelledOutcome"].sum()),
        "positive_test": int(test["CancelledOutcome"].sum()), "censor_days": censor_days, "cutoff_date": str(cutoff.date()),
    })
    return model, metrics, labeled, test


def score_cancellation_risk(model_bundle, orders: pd.DataFrame) -> pd.DataFrame:
    if model_bundle is None:
        return orders.assign(CancellationProbability=np.nan, CancellationRisk="Unavailable")
    model = model_bundle[0]
    x = orders.copy()
    numeric = CANCEL_FEATURES_NUMERIC
    categorical = CANCEL_FEATURES_CATEGORICAL
    x["CancellationProbability"] = model.predict_proba(x[numeric + categorical])[:, 1]
    x["CancellationRisk"] = pd.cut(x["CancellationProbability"], bins=[-0.01, 0.20, 0.50, 1.01], labels=["Low", "Medium", "High"])
    return x


def kpis(df: pd.DataFrame, match_df: pd.DataFrame | None = None) -> dict:
    orders = order_level(df)
    metrics = cancellation_metrics(df, match_df)
    gross_sales = sales_lines(df)["Revenue"].sum()
    returns = df.loc[df["IsReturnAdjustment"], "ReturnValue"].sum()
    net_sales = gross_sales - returns
    customers = sales_lines(df)["CustomerID"].nunique(dropna=True)
    cm = customer_metrics(df)
    repeat_rate = float((cm["Orders"] > 1).mean() * 100) if len(cm) else 0
    return {
        "orders": len(orders), "gross_revenue": float(gross_sales), "return_value": float(returns), "net_revenue": float(net_sales),
        "customers": int(customers), "cancel_rate": metrics["matched_cancellation_rate"], "raw_cancel_rate": metrics["raw_cancellation_invoice_rate"],
        "aov": float(gross_sales / max(1, len(orders))), "repeat_rate": repeat_rate,
    }
