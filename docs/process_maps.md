# Order IQ — Process Maps

## Observed historical transaction process

Customer order → sales invoice → possible cancellation/return event → transaction record.

The UCI dataset does **not** contain inventory levels, payment events, warehouse queue timestamps, or fulfillment completion timestamps. Those are not presented as observed facts.

## Measured signals

- Matched cancellation rate: likely cancelled original sales orders / genuine positive sales orders.
- Cancellation lag: matched cancellation date − original order date.
- Hourly and weekday order volume: transaction load proxies.
- Product cancellation hotspots: matched cancelled original orders / product orders, with a minimum order-count filter.

## Proposed future-state workflow

New order → validation → cancellation-risk prediction → anomaly/risk scoring → priority assignment → fulfillment queue → completion.

Inventory/availability checking is shown as a proposed control, not as a measured historical stage.
