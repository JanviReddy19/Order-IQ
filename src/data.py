from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_PATH = DATA_DIR / "demo_orders.csv"

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
]
SERVICE_CODES = {"POST", "M", "D", "DOT", "BANK CHARGES", "AMAZONFEE", "C2"}
SERVICE_TERMS = ("POSTAGE", "BANK CHARGE", "AMAZON FEE", "DOTCOM", "MANUAL")


class DataQualityError(ValueError):
    """Raised when a loaded dataset is structurally unusable."""


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Do not silently manufacture key columns. Missing keys are a data-quality error.
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[REQUIRED_COLUMNS]


def _validate_loaded_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataQualityError("Loaded dataset is empty")
    invoice_missing = df["InvoiceNo"].isna().mean()
    if invoice_missing > 0.50:
        raise DataQualityError(
            f"InvoiceNo is {invoice_missing:.1%} missing; refusing to continue with a collapsed order key"
        )
    if df["InvoiceNo"].astype(str).str.strip().replace("", np.nan).notna().mean() < 0.50:
        raise DataQualityError("InvoiceNo contains too few usable identifiers")
    if df["InvoiceNo"].astype(str).nunique() < 2:
        raise DataQualityError("InvoiceNo has fewer than two unique values")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    _validate_loaded_frame(df)

    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
    df["Description"] = df["Description"].fillna("Unknown product").astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=["InvoiceNo", "Quantity", "UnitPrice", "InvoiceDate"])
    df = df[df["UnitPrice"] >= 0]

    df["IsCancellationInvoice"] = df["InvoiceNo"].str.upper().str.startswith("C")
    # Negative non-C rows are treated as returns/adjustments, not cancellations.
    df["IsReturnAdjustment"] = (~df["IsCancellationInvoice"]) & (df["Quantity"] < 0)
    df["IsCancelled"] = df["IsCancellationInvoice"]  # compatibility alias
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["GrossSales"] = np.where(df["Quantity"] > 0, df["Revenue"], 0.0)
    df["ReturnValue"] = np.where(df["Quantity"] < 0, -df["Revenue"], 0.0)
    df["OrderID"] = df["InvoiceNo"].str.replace("^C", "", regex=True)
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["Hour"] = df["InvoiceDate"].dt.hour
    code = df["StockCode"].fillna("").astype("string").str.strip().str.upper()
    desc = df["Description"].fillna("").astype("string").str.strip().str.upper()
    service_code_mask = code.isin(SERVICE_CODES)
    service_term_mask = pd.Series(False, index=df.index)
    for term in SERVICE_TERMS:
        service_term_mask |= desc.str.contains(term, regex=False, na=False)
    df["IsProduct"] = code.ne("") & ~service_code_mask & ~service_term_mask
    return df.reset_index(drop=True)


def _is_product_row(row: pd.Series) -> bool:
    code = str(row.get("StockCode", "")).strip().upper()
    desc = str(row.get("Description", "")).strip().upper()
    if code in SERVICE_CODES:
        return False
    if any(term in desc for term in SERVICE_TERMS):
        return False
    return bool(code)


def make_demo_data(n_orders: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic retail dataset with matched cancellations and returns."""
    rng = np.random.default_rng(seed)
    products = [
        ("P100", "Gift Box", 4.5), ("P101", "Ceramic Mug", 7.2),
        ("P102", "Notebook", 3.8), ("P103", "Wall Clock", 12.5),
        ("P104", "Candle Set", 9.9), ("P105", "Tote Bag", 6.5),
        ("P106", "Glass Jar", 5.5), ("P107", "Photo Frame", 8.2),
        ("P108", "Tea Set", 18.0), ("P109", "Desk Organizer", 11.0),
        ("P110", "Gift Wrap", 2.4), ("P111", "Serving Bowl", 14.5),
    ]
    countries = ["India", "United Kingdom", "Germany", "France", "Netherlands", "Spain", "Australia"]
    customer_ids = np.arange(12000, 12700)
    rows: list[list] = []
    base = pd.Timestamp("2010-12-01 08:00")

    # A small set of high-frequency customers makes repeat-customer analysis meaningful.
    weights = np.ones(len(customer_ids))
    weights[:80] = 4.0
    weights /= weights.sum()

    for i in range(n_orders):
        invoice = f"{100000+i}"
        customer = int(rng.choice(customer_ids, p=weights))
        date = base + pd.Timedelta(days=int(rng.integers(0, 370)), hours=int(rng.integers(0, 12)))
        n_lines = int(rng.integers(1, 5))
        chosen = rng.choice(len(products), size=n_lines, replace=False)
        for p_idx in chosen:
            code, desc, base_price = products[int(p_idx)]
            qty = int(max(1, rng.lognormal(1.2, 0.75)))
            if rng.random() < 0.012:
                qty *= int(rng.integers(15, 45))
            price = round(base_price * float(rng.uniform(0.9, 1.15)), 2)
            rows.append([invoice, code, desc, qty, date, price, customer, rng.choice(countries)])

    sales = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    # Match a meaningful subset of cancellation invoices to earlier sales invoices.
    sales_invoices = sales["InvoiceNo"].unique()
    cancel_count = max(1, int(len(sales_invoices) * 0.075))
    cancel_invoices = rng.choice(sales_invoices, size=cancel_count, replace=False)
    cancel_rows = []
    for original in cancel_invoices:
        basket = sales[sales["InvoiceNo"] == original]
        c_invoice = "C" + str(300000 + len(cancel_rows) + 1)
        lag_days = int(rng.integers(1, 25))
        cancel_date = basket["InvoiceDate"].iloc[0] + pd.Timedelta(days=lag_days)
        for _, r in basket.iterrows():
            cancel_rows.append([
                c_invoice, r["StockCode"], r["Description"], -abs(r["Quantity"]),
                cancel_date, r["UnitPrice"], r["CustomerID"], r["Country"],
            ])

    # Add a smaller pool of non-C returns/adjustments to keep the distinction visible.
    return_rows = []
    for _, r in sales.sample(frac=0.025, random_state=seed).iterrows():
        return_rows.append([
            r["InvoiceNo"], r["StockCode"], r["Description"], -int(max(1, r["Quantity"] // 2)),
            r["InvoiceDate"] + pd.Timedelta(days=int(rng.integers(1, 20))),
            r["UnitPrice"], r["CustomerID"], r["Country"],
        ])

    all_rows = pd.concat([
        sales,
        pd.DataFrame(cancel_rows, columns=REQUIRED_COLUMNS),
        pd.DataFrame(return_rows, columns=REQUIRED_COLUMNS),
    ], ignore_index=True)
    return clean_data(all_rows)


def load_data(use_demo: bool = False) -> tuple[pd.DataFrame, str]:
    if use_demo:
        return make_demo_data(), "Synthetic demo dataset (matched cancellations + returns)"

    candidates = list(RAW_DIR.glob("*.xlsx")) + list(PROCESSED_DIR.glob("*.csv"))
    if candidates:
        path = candidates[0]
        try:
            df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
            return clean_data(df), f"Local file: {path.name}"
        except Exception as exc:
            return make_demo_data(), f"Demo fallback (local data quality error: {type(exc).__name__})"

    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=352)
        # ucimlrepo separates ID columns from feature columns. The UCI package
        # explicitly exposes data.ids and data.features as separate dataframes.
        print("UCI feature columns:", list(ds.data.features.columns))
        print("UCI id columns:", list(ds.data.ids.columns) if ds.data.ids is not None else [])
        ids = ds.data.ids if ds.data.ids is not None else pd.DataFrame(index=ds.data.features.index)
        df = pd.concat([ids.reset_index(drop=True), ds.data.features.reset_index(drop=True)], axis=1)
        # Guard against a provider version returning the same variable in both frames.
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        _validate_loaded_frame(df)
        return clean_data(df), "UCI Online Retail (downloaded at runtime)"
    except Exception as exc:
        return make_demo_data(), f"Demo fallback (UCI unavailable/invalid: {type(exc).__name__})"
