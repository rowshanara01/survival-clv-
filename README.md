# Survival Analysis + CLV (REAL Telco data)

Churn classifier bole "chhere jabe kina". Survival analysis bole **"KOBE"** chhere jabe, ar **kotodin** ar ekhane thakbe.
Ekhane "ekhono chhere jai ni" customer ke bad dewa jay na: tara **censored** (jante pari shudhu "oto din porjonto chhilo").

## Chalanor niyom
pip install -r requirements.txt
python train.py          # KM -> Cox -> PH test -> stratify -> calibration -> CLV -> retention policy (+ reports/*.png)
pytest -q
uvicorn app.main:app --reload     # POST /survival  (/docs)

## Ja ja ache (high-level)
1. Censoring: 73% customer ekhono ache, classifier hole eder 'no churn' dhore vul shikhto
2. Kaplan-Meier + log-rank test; Cox Proportional Hazards, hazard ratio (jemon Electronic check = 1.51x jhuki)
3. Leakage: TotalCharges ~ tenure x bill, mane duration er answer-i bole dey, tai bad
4. Proportional-hazards assumption test (Schoenfeld) + remedy: Contract ke STRATA kora
5. Concordance index (test), calibration at 24 months censoring-aware (predicted vs Kaplan-Meier)
6. Survival-based CLV: sum( P(thakbe) x margin x bill x discount ), conditional on current tenure
7. Retention policy: offer dile hazard komle CLV koto bare vs offer cost, positive hole-i offer dao
8. Library gotcha: lifelines stratified predict e column order shuffle kore (bug dhora hoyechhe, test e)
9. API + tests + quality gates + Docker

## Real data theke shikhar kotha
- Two-year contract e prothom 24 mash KEU chhere na (contract-e bandha). Survival curve data-r NIYOM-o dekhay.
- Month-to-month e 24 mash e matro ~59% thake.

## Shimabodhdhota
- Offer er prabhab (hazard x0.8) ASSUMPTION. Asol-e A/B test lage.
- Data max 72 mash; er porer extrapolation kora hoy ni, tai porer 24 mash dekhate pare na jodi tenure beshi hoy.
- Stratified model e kichu streaming feature e PH assumption ekhono 100% thik na (train.py output dekho).

## Challenge
1. Time-varying covariate (jemon contract bodlano), Cox with time-varying
2. Random Survival Forest / GBM survival (scikit-survival)
3. Competing risks: customer "chhere gese" vs "contract upgrade" (alada event)
4. Uplift-coupon project er A/B data diye offer effect shikho
5. Bootstrap diye CLV er confidence interval
