# Survival Analysis + CLV (Real Telco Data) #

A churn classifier tells you *whether* a customer will leave. Survival analysis tells you **when** they will leave, and **how long** they will stay.

Customers who haven't left yet can't simply be dropped: they are **censored**. All we know is "they were still here after this many months."

## How to run

```bash
pip install -r requirements.txt
python train.py                   # KM -> Cox -> PH test -> stratify -> calibration -> CLV -> retention policy (+ reports/*.png)
pytest -q
uvicorn app.main:app --reload     # POST /survival  (docs at /docs)
```

## What's included (high level) ##

1. **Censoring:** 73% of customers are still active. A classifier would wrongly learn these as "no churn."
2. **Kaplan-Meier + log-rank test; Cox Proportional Hazards** with hazard ratios (for example, Electronic check = 1.51x the risk).
3. **Leakage:** `TotalCharges` is roughly tenure x bill, so it gives away the duration (the answer). It is dropped.
4. **Proportional-hazards assumption test (Schoenfeld)** plus a remedy: `Contract` is used as a **strata** variable.
5. **Concordance index (test set)** and **calibration at 24 months**, censoring-aware (predicted vs Kaplan-Meier).
6. **Survival-based CLV:** `sum( P(still here) x margin x bill x discount )`, conditional on the customer's current tenure.
7. **Retention policy:** if an offer lowers the hazard, compute how much CLV rises versus the offer cost. If the gain is positive, make the offer.
8. **Library gotcha:** `lifelines` shuffles the column order in stratified `predict`. The bug was caught and is covered by a test.
9. **API + tests + quality gates + Docker.**

## What the real data teaches ##

- On two-year contracts, **nobody** leaves in the first 24 months (they are locked in). The survival curve even shows the rules of the data.
- On month-to-month contracts, only about 59% remain at 24 months.

## Limitations

- **The offer effect (hazard x 0.8) is an assumption.** In practice you need an A/B test to measure it.
- The data covers at most 72 months. No extrapolation beyond that was done, so the model can't show the next 24 months for customers with long tenure.
- In the stratified model, the proportional-hazards assumption still isn't fully satisfied for some streaming features (see the `train.py` output).

## Challenges / next steps

1. **Time-varying covariates** (for example, a contract change), using Cox with time-varying covariates.
2. **Random Survival Forest / GBM survival** (`scikit-survival`).
3. **Competing risks:** "customer left" vs "contract upgrade" as separate events.
4. Learn the real offer effect from the A/B data in the **uplift-coupon** project.
5. **Bootstrap confidence intervals** for CLV.
