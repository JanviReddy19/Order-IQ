# Order IQ — Operations Exception Playbook

## Purpose
Use Order IQ as a **review-prioritization aid**, not as an automatic cancellation decision engine.

## Daily workflow

1. **Load new orders** into the order-risk view.
2. **Sort by cancellation probability / composite risk.**
3. Review the highest-risk queue first.
4. Assign an **owner**, set a **status**, and record the **reason**.
5. Check product/customer context before taking an action.
6. Route approved orders to the standard workflow; hold only when an actual operational rule supports it.
7. At the end of the pilot period, compare observed cancellations with the model-ranked review queue.

## Suggested status values

- New
- In Review
- Approved
- Hold
- Closed

## Review rule

The model should prioritize human attention. It should **not** be treated as proof that an order will cancel.

## Pilot measurement

Track:

- orders reviewed
- cancellations caught
- cancellations missed
- review time per order
- reviews per cancellation caught
- customer/product concentration
- false-positive operational cost

Use the 5% / 10% / 20% / 30% pilot-impact table in Order IQ to select a review-capacity scenario, then validate it against live operational cost and outcomes.
