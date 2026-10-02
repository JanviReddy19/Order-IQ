# Order IQ: E-commerce Order Cancellation Analysis and Exception Workflow 📦

> **Operations problem:** e-commerce teams can see orders, cancellations and demand patterns after the fact, but without a structured exception workflow they may review risky orders too late or spend review capacity uniformly. **Order IQ** turns historical transaction data into measured cancellation insights, a risk-ranked exception queue, and a proposed review workflow.

**Live demo:** [Order IQ — Live Demo](https://order-iq-fh7nbrrnmj768bchpd7rsl.streamlit.app/)

**Dataset:** UCI Online Retail contains 541,909 source transaction lines covering 1 Dec 2010–9 Dec 2011. UCI documents `C`-prefixed invoice numbers as cancellation records.

## Real-data findings

The analysis below was generated from the real UCI Online Retail data, not synthetic demo data.

- **Transaction lines:** 541,907 loaded transaction lines
- **Data-cleaning difference:** 2 source rows were excluded during validation/cleaning, so the application reports 541,907 loaded transaction lines.
- **Genuine sales orders:** 20,726
- **Matched cancellation-to-order pairs:** 1,296
- **Matched cancellation rate:** 5.824% of genuine sales orders (**lower-bound estimate**)
- **Median cancellation lag:** 9.99 days
- **Peak order-volume hour:** 12:00
- **Top-10% cancellation precision:** 15.58%
- **Top-10% cancellation recall:** 26.46%
- **Top-10% lift:** 2.13× over the training cancellation base rate
- **PR-AUC:** 0.1256
- **ROC-AUC:** 0.6606

The supervised evaluation uses identified customers only, a chronological split, and a 28-day censoring window. The model metrics are historical back-testing results and should not be interpreted as guaranteed future performance.

## Findings and recommendations

The app generates these findings dynamically from the loaded real dataset.

| Finding | Action |
|---|---|
| Matched cancellation signal | Use it as a lower-bound cancellation signal and focus review capacity on orders with the highest predicted risk. |
| Cancellation timing | The median matched cancellation lag is about 10 days; use it to frame an initial review window, then validate the SLA with live operational data. |
| Product hotspots | Investigate high-rate products only after applying the minimum-volume filter; review product-specific cancellation causes before creating product-specific rules. |
| Peak order hour | Align exception-review and operations coverage with measured order-load peaks rather than staffing uniformly across the day. |
| Top-decile model performance | Use the pilot-impact table to evaluate the trade-off between review workload and cancellations caught. |

## Pilot impact

The held-out test set contains 3,203 eligible orders and 189 observed cancellation outcomes. The risk-ranked review trade-off is:

| Review share | Orders reviewed | Cancellations caught | % of all cancellations caught | Reviews per cancellation caught |
|---:|---:|---:|---:|---:|
| 5% | 160 | 29 | 15.3% | 5.5 |
| 10% | 320 | 50 | 26.5% | 6.4 |
| 20% | 640 | 75 | 39.7% | 8.5 |
| 30% | 960 | 96 | 50.8% | 10.0 |

This is historical back-testing. It does not establish that the same capture rate will hold after deployment.

## Process change

### As-is

![As-is process](docs/as_is_process.png)

### To-be

![To-be process](docs/to_be_process.png)

The to-be workflow adds two explicit controls that are absent from the transaction history: **early exception screening** and **risk-based review**. The dataset does not contain inventory, payment or warehouse completion timestamps, so those stages are proposed controls rather than measured historical facts.

## Screenshots

### Executive Dashboard

![Executive Dashboard](docs/screenshots/01-dashboard.png)

### Findings & Recommendations

![Findings & Recommendations](docs/screenshots/02-findings.png)

### ML & Risk

![ML & Risk](docs/screenshots/03-ml-risk.png)

### Pilot Impact

![Pilot Impact](docs/screenshots/04-pilot-impact.png)

### Process Workflow

![Process Workflow](docs/screenshots/05-process.png)

## What the system does

1. Loads UCI data using `ucimlrepo.data.ids + data.features` and validates `InvoiceNo` before processing.
2. Separates genuine positive sales, `C`-invoice cancellation records and negative non-C returns/adjustments.
3. Matches cancellation lines to likely earlier positive orders using identified customer, product, earlier date and exact quantity.
4. Reports **Matched cancellation rate**, explicitly as a lower bound because anonymous, unmatched and partial-quantity cases are excluded.
5. Gives anonymous orders `IsAnonymous=True` and zero customer history rather than grouping all anonymous customers together.
6. Adds leakage-safe `PriorOrders`, `PriorRevenue`, `PriorCancellations` and `PriorCancellationRate`.
7. Measures cancellation lag, repeat-purchase gaps, hourly and weekday order load.
8. Uses Isolation Forest for outcome-free anomaly detection.
9. Uses a chronological Random Forest cancellation model on identified customers only; the last 28 days are censored from evaluation to reduce right-censoring.
10. Reports top-decile precision/recall, PR-AUC, ROC-AUC and lift against the training cancellation base rate.
11. Converts model predictions into a **pilot impact table**: if operations reviews the riskiest 5/10/20/30% of orders, how many cancellations are caught and how many reviews are required per catch.
12. Provides an **exceptions tracker** with owner, status and reason, downloadable as CSV.
13. Provides a country selector in the order simulator and reuses fitted models/reference distributions rather than refitting on each click.

## ML and operations logic

### Isolation Forest

The anomaly model is trained on genuine positive sales orders only. Cancellation/return outcomes are never fed into the anomaly score.

### Cancellation model

A Random Forest estimates cancellation probability from information available at order time: order value, basket size, customer history, prior cancellation history, hour, weekday and country. The headline evaluation uses a chronological split and no arbitrary 0.5 threshold.

### Pilot impact

The operational question is:

> **If the team reviews the riskiest 10% of orders, what percentage of observed cancellations would be caught, and how many reviews are needed per cancellation caught?**

This is back-testing on historical data, not proof of future operational performance.

## Currency

The source dataset is in GBP. Order IQ keeps GBP internally and may show INR using a clearly labelled fixed presentation reference. This is cosmetic and **not** a 2010–11 historical purchasing-power conversion.

## Limitations

- UCI has no inventory levels, payment events, warehouse queue timestamps or fulfillment completion timestamps; the proposed workflow cannot claim to predict fulfillment performance.
- Cancellation matching is heuristic and exact-quantity based, so matched cancellation rate is a lower bound.
- Anonymous orders cannot be reliably linked to customer-level cancellation outcomes and are excluded from supervised model evaluation.
- The last 28 days are censored from supervised evaluation because their cancellation outcomes may not yet be fully observed.
- Historical 2010–11 results describe this dataset and should not be presented as current e-commerce benchmarks.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For development/testing:

```bash
pip install -r requirements-dev.txt
pytest -q
```

To generate the real-data metrics report locally after the UCI dataset is available:

```bash
python scripts/run_real_report.py
```

The script intentionally fails if the loader falls back to synthetic data, preventing accidental publication of demo metrics.

## Deployment

Streamlit Community Cloud deploys this repository using `app.py` as the entrypoint. The live application is linked above.

## Dataset citation

Chen, D. (2015). **Online Retail**. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
