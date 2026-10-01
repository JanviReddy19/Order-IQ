from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from data import load_data
from analytics import order_level, match_cancellations, cancellation_metrics, product_cancellation_hotspots, fit_cancellation_model


def main():
    df, source = load_data(False)
    if source.startswith('Demo fallback'):
        raise RuntimeError(f'Real-data run failed and fell back to demo mode: {source}')
    orders = order_level(df)
    match_df, _ = match_cancellations(df)
    cm = cancellation_metrics(df, match_df)
    hotspots = product_cancellation_hotspots(df, match_df, min_orders=20).head(10)
    hourly = orders.groupby('Hour').size().sort_values(ascending=False)
    model_bundle = fit_cancellation_model(
        orders, match_df,
        cancellation_dates=df.loc[df['IsCancellationInvoice'], 'InvoiceDate'],
        censor_days=28,
    )
    out = {
        'source': source,
        'transaction_lines': int(len(df)),
        'sales_orders': int(len(orders)),
        'matched_cancellation_rate_pct': round(cm['matched_cancellation_rate'], 3),
        'cancellation_invoice_match_rate_pct': round(cm['cancellation_invoice_match_rate'], 3),
        'peak_order_hour': int(hourly.index[0]) if len(hourly) else None,
        'top_hotspot_products': hotspots[['StockCode','Product','Orders','CancelledOrders','CancellationRate']].to_dict('records'),
    }
    if model_bundle is not None:
        _, metrics, _, _ = model_bundle
        out['model'] = metrics
    else:
        out['model'] = None

    Path('reports').mkdir(exist_ok=True)
    Path('reports/real_metrics.json').write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
