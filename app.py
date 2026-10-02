from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from data import load_data
from analytics import (
    order_level, match_cancellations, cancellation_metrics,
    product_cancellation_hotspots, risk_engine,
    fit_anomaly_model, fit_cancellation_model, score_cancellation_risk,
    kpis, sales_lines,
)
from metrics import pilot_impact_table

GBP_TO_INR = 127.30
RATE_DATE = "2026-10-01"

st.set_page_config(page_title="Order IQ", page_icon="📦", layout="wide")
st.title("📦 Order IQ: E-commerce Order Cancellation Analysis and Exception Workflow")
st.caption("Operations analytics → measured findings → exception workflow → ML-enabled review prioritization")
st.caption(f"Source currency: GBP. INR is a presentation conversion only: £1 = ₹{GBP_TO_INR:.2f} (reference date {RATE_DATE}).")

with st.sidebar:
    st.header("Controls")
    use_demo = st.checkbox(
        "Use improved demo data", value=False,
        help="Synthetic data is for offline testing only. Never use demo-mode metrics on a resume."
    )
    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Validate source data\n2. Match cancellations\n3. Build leakage-safe history\n4. Measure bottlenecks\n5. Predict cancellation risk\n6. Run an exception-review pilot")

@st.cache_data(show_spinner="Loading and validating transaction data...")
def get_data(demo: bool):
    return load_data(demo)

@st.cache_data(show_spinner="Building order-level process dataset...")
def get_analysis(demo: bool, source: str):
    df, _ = get_data(demo)
    orders = order_level(df)
    match_df, line_matches = match_cancellations(df)
    cancel_metrics = cancellation_metrics(df, match_df)
    hotspots = product_cancellation_hotspots(df, match_df, min_orders=20)
    return orders, match_df, line_matches, cancel_metrics, hotspots

@st.cache_resource(show_spinner="Fitting anomaly model once for this dataset...")
def get_anomaly_bundle(demo: bool, source: str):
    orders, *_ = get_analysis(demo, source)
    return fit_anomaly_model(orders)

@st.cache_resource(show_spinner="Training cancellation model once for this dataset...")
def get_cancellation_bundle(demo: bool, source: str):
    df, _ = get_data(demo)
    orders, match_df, *_ = get_analysis(demo, source)
    cancellation_dates = df.loc[df["IsCancellationInvoice"], "InvoiceDate"]
    return fit_cancellation_model(orders, match_df, cancellation_dates=cancellation_dates, censor_days=28)


df, source = get_data(use_demo)
orders, match_df, line_matches, cancel_metrics, hotspots = get_analysis(use_demo, source)
anomaly_bundle = get_anomaly_bundle(use_demo, source)
orders_scored = risk_engine(orders, anomaly_bundle)
cancel_model_bundle = get_cancellation_bundle(use_demo, source)
orders_scored = score_cancellation_risk(cancel_model_bundle, orders_scored)
metrics = kpis(df, match_df)

if use_demo:
    st.warning("⚠️ DEMO MODE: synthetic data. Cancellation, hotspot and model metrics are illustrative only and must NOT be presented as real-data results.")
else:
    st.success("REAL-DATA MODE: metrics shown below are calculated from the loaded UCI Online Retail dataset.")

st.info(f"Data source: **{source}** | {len(df):,} transaction lines | {len(orders_scored):,} genuine sales orders | {len(match_df):,} cancellation-to-order matches")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Sales orders", f"{metrics['orders']:,}")
c2.metric("Net revenue", f"₹{metrics['net_revenue'] * GBP_TO_INR:,.0f}")
c3.metric("Identified customers", f"{metrics['customers']:,}")
c4.metric("Matched cancellation rate", f"{metrics['cancel_rate']:.1f}%")
c5.metric("Gross AOV", f"₹{metrics['aov'] * GBP_TO_INR:,.2f}")
c6.metric("Repeat customers", f"{metrics['repeat_rate']:.1f}%")

st.divider()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Dashboard", "Findings & Recommendations", "Process Analysis", "ML & Risk", "Exceptions & Simulator", "Data Quality"
])

sales = sales_lines(df)

with tab1:
    left, right = st.columns(2)
    with left:
        max_month = sales["InvoiceDate"].max().to_period("M") if not sales.empty else None
        monthly = sales.assign(MonthPeriod=sales["InvoiceDate"].dt.to_period("M"))
        if max_month is not None:
            monthly = monthly.loc[monthly["MonthPeriod"] < max_month]
        monthly = monthly.groupby("MonthPeriod", as_index=False)["GrossSales"].sum()
        monthly["GrossSalesINR"] = monthly["GrossSales"] * GBP_TO_INR
        monthly["Month"] = monthly["MonthPeriod"].astype(str)
        st.plotly_chart(px.line(monthly, x="Month", y="GrossSalesINR", markers=True, title="Monthly Gross Sales — Complete Months Only"), use_container_width=True)
        st.caption("The final source month is partial and is excluded from this trend.")
    with right:
        country = sales.groupby("Country", as_index=False)["GrossSales"].sum().sort_values("GrossSales", ascending=False).head(10)
        country["GrossSalesINR"] = country["GrossSales"] * GBP_TO_INR
        st.plotly_chart(px.bar(country.sort_values("GrossSalesINR"), x="GrossSalesINR", y="Country", orientation="h", title="Top Countries by Gross Sales"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        top = sales.loc[sales["IsProduct"]].groupby("Description", as_index=False)["GrossSales"].sum().sort_values("GrossSales", ascending=False).head(10)
        top["GrossSalesINR"] = top["GrossSales"] * GBP_TO_INR
        st.plotly_chart(px.bar(top.sort_values("GrossSalesINR"), x="GrossSalesINR", y="Description", orientation="h", title="Top Products by Gross Sales"), use_container_width=True)
    with right:
        risk_counts = orders_scored["RiskLevel"].value_counts().rename_axis("Risk").reset_index(name="Orders")
        st.plotly_chart(px.pie(risk_counts, names="Risk", values="Orders", title="New-Order Risk Distribution"), use_container_width=True)

    st.subheader("Warehouse / operations load signals")
    left, right = st.columns(2)
    hourly = sales.groupby("Hour").size().reset_index(name="Orders")
    with left:
        st.plotly_chart(px.bar(hourly, x="Hour", y="Orders", title="Order Volume by Hour"), use_container_width=True)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = sales["DayOfWeek"].value_counts().reindex(weekday_order).fillna(0).reset_index()
    weekday.columns = ["Day", "Orders"]
    with right:
        st.plotly_chart(px.bar(weekday, x="Day", y="Orders", title="Order Volume by Weekday"), use_container_width=True)

with tab2:
    st.subheader("Findings and Recommendations")
    st.caption("These statements are generated from the loaded dataset. In real-data mode they are suitable for your project report; in demo mode they are explicitly illustrative.")

    # Build cancellation outcome for descriptive findings only.
    finding_orders = orders.copy()
    cancelled_originals = set(match_df["OriginalInvoice"].dropna())
    finding_orders["CancelledOutcome"] = finding_orders["InvoiceNo"].isin(cancelled_originals).astype(int)

    # 1. Matched cancellation rate
    st.markdown(f"**1. Matched cancellation signal — {cancel_metrics['matched_cancellation_rate']:.1f}%** of genuine sales orders are linked to a cancellation record under the matching rule. **Action:** use this as the baseline for a targeted exception-review workflow, while treating it as a lower bound.")

    # 2. Cancellation lag
    lag_days = None
    if not match_df.empty:
        md = match_df.merge(orders[["InvoiceNo", "InvoiceDate"]].rename(columns={"InvoiceNo":"OriginalInvoice", "InvoiceDate":"OriginalDate"}), on="OriginalInvoice", how="left")
        cd = df.loc[df["IsCancellationInvoice"], ["InvoiceNo", "InvoiceDate"]].drop_duplicates().rename(columns={"InvoiceNo":"CancellationInvoice", "InvoiceDate":"CancellationDate"})
        lag = md.merge(cd, on="CancellationInvoice", how="left")
        lag["LagDays"] = (lag["CancellationDate"] - lag["OriginalDate"]).dt.total_seconds() / 86400
        lag = lag.dropna(subset=["LagDays"])
        if not lag.empty:
            lag_days = float(lag["LagDays"].median())
    if lag_days is not None:
        st.markdown(f"**2. Cancellation timing — median lag is {lag_days:.1f} days.** **Action:** start an exception-review window before this lag and validate the chosen SLA with live operations data.")
    else:
        st.markdown("**2. Cancellation timing — no matched lag can be estimated.** **Action:** improve cancellation matching coverage before setting an operational review window.")

    # 3. Hotspot product
    if not hotspots.empty:
        h = hotspots.iloc[0]
        share = float(h["CancelledOrders"] / max(cancel_metrics["matched_original_orders"], 1) * 100)
        st.markdown(f"**3. Product hotspot — {h['Product']} has a {h['CancellationRate']:.1f}% matched cancellation rate across {int(h['Orders']):,} orders; its matched cancelled orders represent {share:.1f}% of matched cancelled original orders.** **Action:** review product-level cancellation reasons, availability and merchandising conditions before applying a product-specific rule.")
    else:
        st.markdown("**3. Product hotspot — no product meets the minimum-volume threshold.** **Action:** collect more observations before creating product-specific exception rules.")

    # 4. Peak hour and cancellation rate by hour
    hourly_order = finding_orders.groupby("Hour").agg(Orders=("InvoiceNo", "nunique"), Cancelled=("CancelledOutcome", "sum")).reset_index()
    hourly_order["CancellationRate"] = hourly_order["Cancelled"] / hourly_order["Orders"].clip(lower=1) * 100
    peak = hourly_order.sort_values("Orders", ascending=False).iloc[0]
    st.markdown(f"**4. Load peak — hour {int(peak['Hour']):02d}:00 has the highest sales-order volume ({int(peak['Orders']):,} orders).** **Action:** align exception-review/operations coverage with the measured peak rather than staffing uniformly across the day.")

    # 5. Model / pilot
    if cancel_model_bundle is not None:
        _, cm, _, test = cancel_model_bundle
        st.markdown(f"**5. Cancellation-risk model — the held-out top 10% has {cm['top10_precision']:.1%} precision, {cm['top10_recall']:.1%} recall and {cm['top10_lift']:.2f}× lift over the {cm['baseline_rate']:.1%} training base rate.** **Action:** use the pilot-impact table below to decide whether the review workload is operationally worthwhile.")
    else:
        st.markdown("**5. Cancellation-risk model — insufficient real-data labels for evaluation.** **Action:** improve matched cancellation coverage before using model performance to size a review queue.")

with tab3:
    st.subheader("As-is process")
    st.image("docs/as_is_process.png", caption="As-is: historical transaction data records the sale and any later cancellation/return event; operational queue stages are not observed.")
    st.subheader("To-be exception workflow")
    st.image("docs/to_be_process.png", caption="To-be: add early risk screening and an explicit exception-review queue before standard fulfillment.")

    a, b, c, d = st.columns(4)
    a.metric("Cancellation invoices", f"{cancel_metrics['cancellation_invoices']:,}")
    b.metric("Matched original orders", f"{cancel_metrics['matched_original_orders']:,}")
    c.metric("Cancellation invoice match rate", f"{cancel_metrics['cancellation_invoice_match_rate']:.1f}%")
    d.metric("Matched cancellation rate", f"{cancel_metrics['matched_cancellation_rate']:.1f}%")
    st.info("Matched cancellation rate is a lower bound: anonymous, unmatched and partial-quantity cancellations are excluded.")

    st.subheader("Cancellation hotspot products")
    if hotspots.empty:
        st.warning("No product has reached the minimum 20-order threshold.")
    else:
        view = hotspots.head(15).copy()
        view["CancellationRate"] = view["CancellationRate"].round(1)
        st.dataframe(view[["StockCode", "Product", "Orders", "CancelledOrders", "CancellationRate"]], use_container_width=True, hide_index=True)

    st.subheader("Measured cancellation lag")
    if lag_days is not None:
        st.metric("Median cancellation lag", f"{lag_days:.1f} days")
        st.caption("Use this as a descriptive signal, not a proven SLA recommendation.")
    else:
        st.warning("No matched cancellation lag available.")

with tab4:
    st.subheader("ML #1 — Isolation Forest anomaly detection")
    a, b, c = st.columns(3)
    a.metric("Orders flagged unusual", f"{orders_scored['IsAnomaly'].mean() * 100:.1f}%")
    b.metric("High-risk new orders", f"{(orders_scored['RiskLevel'] == 'High').sum():,}")
    c.metric("Anomaly/history features", f"{len(anomaly_bundle.feature_names)}")

    st.subheader("ML #2 — Supervised cancellation-risk model")
    if cancel_model_bundle is None:
        st.warning("Not enough identified-customer matched cancellations and sufficiently old labelled orders to train/evaluate the supervised model.")
    else:
        _, cm, labeled, test = cancel_model_bundle
        a, b, c, d, e, f = st.columns(6)
        a.metric("Top-10% precision", f"{cm['top10_precision']:.1%}")
        b.metric("Top-10% recall", f"{cm['top10_recall']:.1%}")
        c.metric("PR-AUC", f"{cm['average_precision']:.3f}")
        d.metric("ROC-AUC", f"{cm['roc_auc']:.3f}")
        e.metric("Top-10% lift", f"{cm['top10_lift']:.2f}×")
        f.metric("Base rate", f"{cm['baseline_rate']:.1%}")
        st.caption(f"Chronological split; identified customers only; final {cm['censor_days']} days censored before {cm['cutoff_date']}. Headline metrics use top-decile ranking, not a 0.5 threshold.")

        model = cancel_model_bundle[0]
        feature_cols = ["OrderValue", "TotalQuantity", "UniqueProducts", "PriorOrders", "PriorRevenue", "DaysSincePriorOrder", "PriorCancellations", "PriorCancellationRate", "Hour", "Weekday", "Country"]
        probs = model.predict_proba(test[feature_cols])[:, 1]
        eval_df = test[["InvoiceNo", "InvoiceDate", "CancelledOutcome", "Country"]].copy()
        eval_df["CancellationProbability"] = probs
        st.plotly_chart(px.histogram(eval_df, x="CancellationProbability", color="CancelledOutcome", nbins=30, title="Held-Out Cancellation Probability Distribution"), use_container_width=True)

        st.subheader("Pilot impact")
        st.write("Operational interpretation: how much of the observed cancellation workload is captured if the team reviews only the riskiest orders?")
        ranked = eval_df.sort_values("CancellationProbability", ascending=False).reset_index(drop=True)
        total = ranked["CancelledOutcome"].sum()
        rows = []
        for share in [0.05, 0.10, 0.20, 0.30]:
            k = max(1, int(len(ranked) * share))
            caught = int(ranked.head(k)["CancelledOutcome"].sum())
            rows.append({
                "Review share": f"{share:.0%}",
                "Orders reviewed": k,
                "Cancellations caught": caught,
                "% of all cancellations caught": round(100 * caught / max(total, 1), 1),
                "Reviews per cancellation caught": round(k / max(caught, 1), 1),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("This is historical back-testing. It does not establish that the same capture rate will hold after deployment.")

    st.subheader("Highest-risk orders — outcome-free scoring")
    high = orders_scored.sort_values("RiskScore", ascending=False).head(20).copy()
    high["Order Value (₹)"] = high["OrderValue"] * GBP_TO_INR
    high["Prior cancellation rate"] = (high["PriorCancellationRate"] * 100).round(1)
    cols = ["InvoiceNo", "Order Value (₹)", "TotalQuantity", "UniqueProducts", "IsAnonymous", "PriorCancellations", "Prior cancellation rate", "AnomalyScore", "RiskScore", "RiskLevel", "Priority", "CancellationProbability"]
    high_display = high[cols].copy()
    # Streamlit renders boolean dataframe cells as checkbox-like controls.
    # Keep the underlying boolean for analytics, but use readable text in the read-only risk table.
    high_display["IsAnonymous"] = high_display["IsAnonymous"].map({True: "Yes", False: "No"})
    st.dataframe(high_display, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Exceptions tracker")
    flagged = orders_scored.sort_values(["RiskScore", "CancellationProbability"], ascending=False).head(100).copy()
    def reason(r):
        reasons = []
        if bool(r.get("IsAnomaly", False)): reasons.append("Unusual order profile")
        if float(r.get("PriorCancellationRate", 0)) > 0: reasons.append("Prior cancellation history")
        if float(r.get("CancellationProbability", 0) or 0) >= 0.5: reasons.append("High predicted cancellation risk")
        if not reasons: reasons.append("High composite risk score")
        return "; ".join(reasons)
    flagged["Reason"] = flagged.apply(reason, axis=1)
    flagged["Owner"] = "Operations Queue"
    flagged["Status"] = "New"
    flagged["Order Value (₹)"] = flagged["OrderValue"] * GBP_TO_INR
    display_cols = ["InvoiceNo", "InvoiceDate", "Country", "Order Value (₹)", "RiskScore", "CancellationProbability", "Priority", "Reason", "Owner", "Status"]
    edited = st.data_editor(flagged[display_cols], use_container_width=True, hide_index=True, num_rows="fixed", column_config={
        "Owner": st.column_config.TextColumn("Owner"),
        "Status": st.column_config.SelectboxColumn("Status", options=["New", "In Review", "Approved", "Hold", "Closed"]),
        "CancellationProbability": st.column_config.NumberColumn("Cancellation probability", format="0.0%"),
        "Order Value (₹)": st.column_config.NumberColumn("Order Value (₹)", format="₹%,.0f"),
    })
    st.download_button("Download exceptions CSV", edited.to_csv(index=False).encode("utf-8"), "order_iq_exceptions.csv", "text/csv")

    st.divider()
    st.subheader("New-order simulator")
    st.write("Uses the same fitted anomaly/reference model and trained cancellation model. No model is refit when you click Evaluate.")
    countries = sorted(set(orders["Country"].dropna().astype(str)))
    a, b, c = st.columns(3)
    value_inr = a.number_input("Order value (₹)", min_value=0.0, value=25000.0, step=1000.0)
    quantity = b.number_input("Total quantity", min_value=1, value=20, step=1)
    products = c.number_input("Unique products", min_value=1, value=3, step=1)
    d, e, f = st.columns(3)
    prior_orders = d.number_input("Customer prior orders", min_value=0, value=2, step=1)
    prior_revenue_inr = e.number_input("Customer prior spend (₹)", min_value=0.0, value=50000.0, step=1000.0)
    prior_cancellations = f.number_input("Customer prior cancellations", min_value=0, value=0, step=1)
    g, h, i = st.columns(3)
    days_since = g.number_input("Days since previous order", min_value=0.0, value=30.0, step=1.0)
    hour = h.slider("Order hour", 0, 23, 12)
    weekday = i.selectbox("Order weekday", list(range(7)), format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    country = st.selectbox("Customer country", countries, index=(countries.index("United Kingdom") if "United Kingdom" in countries else 0))
    if st.button("Evaluate New Order", type="primary"):
        prior_cancel_rate = prior_cancellations / max(prior_orders, 1)
        simulated = pd.DataFrame({
            "InvoiceNo": ["SIMULATED"], "OrderValue": [value_inr / GBP_TO_INR], "TotalQuantity": [quantity], "UniqueProducts": [products], "CustomerID": [999999], "Country": [country],
            "InvoiceDate": [pd.Timestamp("2011-12-01") + pd.Timedelta(hours=hour)], "Hour": [hour], "Weekday": [weekday], "Month": ["2011-12"],
            "PriorOrders": [prior_orders], "PriorRevenue": [prior_revenue_inr / GBP_TO_INR], "DaysSincePriorOrder": [days_since], "PriorCancellations": [prior_cancellations], "PriorCancellationRate": [prior_cancel_rate], "IsAnonymous": [False],
        })
        scored = risk_engine(simulated, anomaly_bundle).iloc[0]
        st.success(f"Priority: **{scored['Priority']}**")
        x, y, z = st.columns(3)
        x.metric("Risk score", f"{scored['RiskScore']:.1f}/100")
        y.metric("Risk level", str(scored['RiskLevel']))
        z.metric("Anomaly score", f"{scored['AnomalyScore']:.1f}/100")
        if cancel_model_bundle is not None:
            scored_cancel = score_cancellation_risk(cancel_model_bundle, simulated).iloc[0]
            st.metric("Predicted cancellation probability", f"{scored_cancel['CancellationProbability']:.1%}")
        st.info("Recommended workflow: route high-risk orders to the exception queue before standard fulfillment.")

with tab6:
    st.subheader("Data-quality controls")
    dq = pd.DataFrame([
        ["UCI loader", "Concatenates ucimlrepo data.ids + data.features and validates InvoiceNo before cleaning."],
        ["Anonymous customers", "IsAnonymous is explicit; anonymous orders receive zero prior customer history."],
        ["Cancellation history", "PriorCancellations and PriorCancellationRate count only identified-customer cancellations strictly before each order."],
        ["Cancellation", "C-prefixed invoices are cancellation records; negative non-C rows are returns/adjustments."],
        ["Cancellation matching", "Closest earlier positive line with same customer, stock code and exact quantity is used as the likely original."],
        ["Matched cancellation rate", "Matched original cancelled orders ÷ genuine positive sales orders; lower bound."],
        ["Product hotspot", "Cancelled original orders ÷ product orders, minimum 20-order threshold and service-code exclusions."],
        ["ML evaluation", "Identified customers only; chronological split; final 28 days censored; top-decile precision/recall, PR-AUC and ROC-AUC."],
        ["Pilot impact", "5/10/20/30% review-share table showing cancellation capture and reviews per catch."],
        ["Exceptions tracker", "Top-risk orders can be assigned owner/status and downloaded as CSV."],
        ["Caching", "Cached functions use stable demo/source keys rather than hashing large DataFrames on every rerun."],
    ], columns=["Control", "Implementation"])
    st.dataframe(dq, use_container_width=True, hide_index=True)

    st.subheader("Dataset fields")
    dictionary = pd.DataFrame([
        ["InvoiceNo", "Transaction/invoice identifier; UCI documents a leading C as cancellation."],
        ["StockCode", "Product identifier."], ["Description", "Product description."], ["Quantity", "Quantity per transaction line."],
        ["InvoiceDate", "Transaction date and time."], ["UnitPrice", "Source unit price in GBP."], ["CustomerID", "Customer identifier; some records are anonymous."], ["Country", "Customer country."],
    ], columns=["Field", "Meaning"])
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

st.divider()
st.caption("Order IQ uses historical UCI Online Retail transactions. It does not claim to predict inventory or fulfillment performance because those operational fields are absent from the source data.")
