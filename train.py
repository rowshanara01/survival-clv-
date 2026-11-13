"""Survival analysis on REAL Telco data: KM -> Cox -> PH assumption check -> stratify -> calibration -> CLV -> retention policy"""
import json
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

import common as C

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df, n0 = C.load()
print(f"customers: {len(df)} (tenure=0 {n0}ta bad) | churned: {df.event.mean():.1%} | censored (ekhono ache): {(1 - df.event.mean()):.1%}")

# ---- 1. Kaplan-Meier: contract onujayi 'bachar' curve ----
fig, ax = plt.subplots(figsize=(7, 4))
km_tab = {}
for k, g in df.groupby("Contract"):
    km = KaplanMeierFitter().fit(g.duration, g.event, label=k)
    km.plot_survival_function(ax=ax, ci_show=False)
    km_tab[k] = {f"S({t}m)": float(km.predict(t)) for t in (12, 24, 48)}
ax.set_xlabel("tenure (months)"); ax.set_ylabel("P(customer ekhono ache)"); ax.set_title("Kaplan-Meier by contract")
fig.tight_layout(); fig.savefig("reports/km_by_contract.png", dpi=110); plt.close(fig)
lr_p = float(multivariate_logrank_test(df.duration, df.Contract, df.event).p_value)
print("\nKaplan-Meier survival by contract:\n", pd.DataFrame(km_tab).T.round(3).to_string(), f"\nlog-rank p-value: {lr_p:.2e}")

# ---- 2. Split + models ----
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.event)
prep = C.Prep().fit(train)
Xtr, Xte = prep.transform(train), prep.transform(test)
mtr = pd.concat([Xtr, train[["duration", "event"]].reset_index(drop=True)], axis=1)


def fit_cox(frame, strata=None):
    return CoxPHFitter(penalizer=0.05).fit(frame, "duration", "event", strata=strata)


# a. contract-only (baseline), b. full, Contract covariate (PH vi-olate hote pare), c. full, Contract STRATA
dummies = pd.get_dummies(train.Contract, drop_first=True).astype(float).reset_index(drop=True)
base_frame = pd.concat([dummies, mtr[["duration", "event"]]], axis=1)
cox_base = fit_cox(base_frame)
full_frame = pd.concat([Xtr.drop(columns=C.STRATA), dummies, mtr[["duration", "event"]]], axis=1)
cox_full = fit_cox(full_frame)
cox_strat = fit_cox(mtr, strata=[C.STRATA])

d_te = pd.get_dummies(test.Contract, drop_first=True).astype(float).reset_index(drop=True)
te_base, te_full = d_te, pd.concat([Xte.drop(columns=C.STRATA), d_te], axis=1)


def cindex(model, Xe):
    s24 = C.surv(model, Xe, [24]).ravel()   # boro S(24) = kom risk
    return float(concordance_index(test.duration.values, s24, test.event.values))


cidx = {"contract_only_cox": cindex(cox_base, te_base), "full_cox_contract_as_covariate": cindex(cox_full, te_full),
        "full_cox_stratified_by_contract": cindex(cox_strat, Xte)}
print("\nTEST concordance index (0.5 = random, 1.0 = perfect):")
for k, v in cidx.items():
    print(f"  {k:34s} {v:.3f}")

# ---- 3. Proportional hazards assumption check (Schoenfeld) ----
def ph_violations(model, frame, strata=None):
    t = proportional_hazard_test(model, frame, time_transform="rank").summary
    bad = t[t.p < 0.01]
    return int(len(bad)), int(len(t)), bad.sort_values("p").index.tolist()[:4]


v_full, v_strat = ph_violations(cox_full, full_frame), ph_violations(cox_strat, mtr)
print(f"\nPH assumption violations (p<0.01): contract-as-covariate {v_full[0]}/{v_full[1]} {v_full[2]}  |  stratified {v_strat[0]}/{v_strat[1]} {v_strat[2]}")

# ---- 4. Calibration at 24 months (censoring-aware: prediction vs Kaplan-Meier) ----
S_te = C.survival_matrix(cox_strat, Xte)
pred24 = S_te[:, 24]
q = pd.qcut(pd.Series(pred24).rank(method="first"), 5, labels=False)
calib = []
for b in range(5):
    m = (q == b).values
    obs = float(KaplanMeierFitter().fit(test.duration.values[m], test.event.values[m]).predict(24))
    calib.append({"risk_group": b + 1, "predicted_S24": float(pred24[m].mean()), "observed_S24": obs, "n": int(m.sum())})
cal = pd.DataFrame(calib)
print("\ncalibration @24 months (risk group 1 = shobcheye jhuki):\n", cal.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], "--", color="gray"); ax.scatter(cal.predicted_S24, cal.observed_S24, s=60)
ax.set_xlabel("predicted S(24m)"); ax.set_ylabel("observed (Kaplan-Meier)"); ax.set_title("Calibration at 24 months")
fig.tight_layout(); fig.savefig("reports/calibration_24m.png", dpi=110); plt.close(fig)

# ---- 5. Survival-based CLV + retention policy (shudhu ACTIVE customer der jonno) ----
active = np.where(test.event.values == 0)[0]
ten, mon = test.tenure.values, test.MonthlyCharges.values
rows = []
for i in active:
    base, k = C.clv(S_te[i], ten[i], mon[i])
    treated, _ = C.clv(S_te[i], ten[i], mon[i], C.ECON["hazard_multiplier"])
    rows.append((i, base, treated - base - C.ECON["offer_cost"], 12 * C.ECON["margin"] * mon[i], k))
pol = pd.DataFrame(rows, columns=["i", "clv", "net_gain", "naive_clv_12m", "horizon"])
treat = pol.net_gain > 0
mtm = (test.Contract.values[pol.i] == "Month-to-month")
policy = {"active_customers": int(len(pol)), "avg_survival_clv_24m": float(pol.clv.mean()),
          "avg_naive_clv_12m": float(pol.naive_clv_12m.mean()),
          "survival_policy": {"treated_share": float(treat.mean()), "expected_net_gain": float(pol.net_gain[treat].sum())},
          "treat_everyone": {"treated_share": 1.0, "expected_net_gain": float(pol.net_gain.sum())},
          "rule_month_to_month": {"treated_share": float(mtm.mean()), "expected_net_gain": float(pol.net_gain[mtm].sum())}}
print(f"\nretention policy on {len(pol)} active test customers (offer prabhab = ASSUMPTION: hazard x{C.ECON['hazard_multiplier']}):")
for k in ("survival_policy", "rule_month_to_month", "treat_everyone"):
    print(f"  {k:22s} treat {policy[k]['treated_share']:.0%}  expected net gain ${policy[k]['expected_net_gain']:.0f}")
print(f"  avg CLV: survival-based 24m ${policy['avg_survival_clv_24m']:.0f} vs naive '12 x bill x margin' ${policy['avg_naive_clv_12m']:.0f}")

# ---- 6. Final model: shob data dekhe retrain (serving) ----
prep_f = C.Prep().fit(df)
Xall = prep_f.transform(df)
cox_final = fit_cox(pd.concat([Xall, df[["duration", "event"]].reset_index(drop=True)], axis=1), strata=[C.STRATA])
joblib.dump({"cph": cox_final, "prep": prep_f}, "model/survival_model.joblib")
coef = cox_final.summary[["exp(coef)"]].rename(columns={"exp(coef)": "hazard_ratio"})
top = pd.concat([coef.nlargest(4, "hazard_ratio"), coef.nsmallest(4, "hazard_ratio")])["hazard_ratio"].round(2).to_dict()
print("\nhazard ratios (>1 beshi churn jhuki, <1 kom):", top)
json.dump({"n_customers": len(df), "km": km_tab, "logrank_p": lr_p, "concordance": cidx,
           "ph_violations": {"contract_as_covariate": list(v_full[:2]), "stratified": list(v_strat[:2])},
           "calibration_24m": calib, "policy": policy, "hazard_ratios": top, "econ": C.ECON},
          open("model/metrics.json", "w"), indent=2)
print("\nsaved model + metrics.json + reports/*.png")
