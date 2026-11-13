"""Shared: data, covariate preparation, survival-based CLV"""
import numpy as np
import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CAT = ["Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", *ADDONS,
       "PaperlessBilling", "PaymentMethod"]          # Contract alada: STRATA hishebe jabe
STRATA = "Contract"
MAX_T = 72                                           # data te max 72 mash dekha gechhe, tar porer jinis extrapolation
ECON = {"margin": 0.35, "horizon": 24, "monthly_discount": 0.01, "hazard_multiplier": 0.80, "offer_cost": 15.0}
# hazard_multiplier 0.80 = offer dile churn er 'hazard' 20% kome (ASSUMPTION, A/B test diye mapte hobe)


def load(path="data/Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    n0 = int((df.tenure == 0).sum())
    df = df[df.tenure > 0].copy()                    # tenure=0 mane duration 0, survival analysis e use kora jay na
    df["duration"] = df.tenure.astype(float)
    df["event"] = (df.Churn == "Yes").astype(int)    # 1 = chhere gese, 0 = ekhono ache (CENSORED)
    return df.reset_index(drop=True), n0


class Prep:
    """Covariate banay. NOTE: TotalCharges ar tenure bad: TotalCharges ~ tenure x monthly, mane duration er answer-i bole dey (leakage)."""
    def _cat(self, df):
        d = df[CAT].copy()
        for c in ADDONS:
            d[c] = d[c].replace("No internet service", "No")   # InternetService=No er sathe perfectly collinear chhilo
        d["MultipleLines"] = d["MultipleLines"].replace("No phone service", "No")
        return d

    def fit(self, df):
        self.cols = list(pd.get_dummies(self._cat(df), drop_first=True).columns)
        self.mu, self.sd = float(df.MonthlyCharges.mean()), float(df.MonthlyCharges.std())
        return self

    def transform(self, df):
        X = pd.get_dummies(self._cat(df)).reindex(columns=self.cols, fill_value=0).astype(float)
        X.index = range(len(df))
        X["MonthlyCharges"] = (df.MonthlyCharges.values - self.mu) / self.sd
        X["SeniorCitizen"] = df.SeniorCitizen.values.astype(float)
        X[STRATA] = df[STRATA].values
        return X


def survival_matrix(cph, X):
    """Prottek customer er jonno S(t), t = 0..72 mash. Shape (customer, 73)"""
    return surv(cph, X, np.arange(0, MAX_T + 1))


def surv(cph, X, times):
    """NOTE: stratified model e lifelines column order shuffle kore (strata onujayi). Tai input er index diye order thik korte hoy."""
    return cph.predict_survival_function(X, times=times).reindex(columns=X.index).T.values


def conditional_curve(S_row, tenure, horizon):
    """Ekhon tenure mash dhore ache, ta dhore porer k mash e thakar probability: S(tenure+k)/S(tenure).
    72 mash er porer data nai, tai oi porjonto-i (horizon chhoto hoye jay)."""
    k = min(horizon, MAX_T - int(tenure))
    if k <= 0:
        return np.array([])
    s0 = max(S_row[int(tenure)], 1e-9)
    return S_row[int(tenure) + 1:int(tenure) + 1 + k] / s0


def clv(S_row, tenure, monthly, hazard_multiplier=1.0, horizon=None):
    """CLV = sum over month: P(thakbe) x margin x monthly bill, discount soho. hazard_multiplier<1 = offer er prabhab"""
    h = horizon or ECON["horizon"]
    cond = conditional_curve(S_row, tenure, h) ** hazard_multiplier
    disc = (1 + ECON["monthly_discount"]) ** -np.arange(1, len(cond) + 1)
    return float(ECON["margin"] * monthly * (cond * disc).sum()), len(cond)
