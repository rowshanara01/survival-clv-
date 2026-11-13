import json

import numpy as np
from fastapi.testclient import TestClient

import common as C
from app.main import app

client = TestClient(app)
M = json.load(open("model/metrics.json"))
HIGH = {"tenure": 2, "monthly_charges": 95, "partner": "No", "dependents": "No", "phone_service": "Yes",
        "multiple_lines": "No", "internet_service": "Fiber optic", "online_security": "No", "online_backup": "No",
        "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "Yes",
        "contract": "Month-to-month", "paperless_billing": "Yes", "payment_method": "Electronic check"}
LOW = {"tenure": 30, "monthly_charges": 45, "partner": "Yes", "dependents": "Yes", "phone_service": "Yes",
       "multiple_lines": "No", "internet_service": "DSL", "online_security": "Yes", "online_backup": "Yes",
       "device_protection": "Yes", "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No",
       "contract": "Two year", "paperless_billing": "No", "payment_method": "Credit card (automatic)"}


def test_censoring_handled_and_no_leakage_columns():
    df, n0 = C.load()
    assert n0 == 11 and 0.70 < 1 - df.event.mean() < 0.76     # ~73% censored (ekhono ache)
    p = C.Prep().fit(df)
    assert not any("TotalCharges" in c or c == "tenure" for c in p.cols)


def test_single_row_transform_equals_batch_transform():
    df, _ = C.load()
    p = C.Prep().fit(df)
    full = p.transform(df)
    for i in (0, 17, 400, 3000, 7000):
        one = p.transform(df.iloc[[i]])
        assert np.allclose(one.drop(columns="Contract").values, full.drop(columns="Contract").iloc[[i]].values)


def test_model_quality_gates():
    c = M["concordance"]
    assert c["full_cox_stratified_by_contract"] >= 0.82
    assert c["full_cox_stratified_by_contract"] > c["contract_only_cox"] + 0.05
    assert all(abs(g["predicted_S24"] - g["observed_S24"]) < 0.08 for g in M["calibration_24m"])
    assert M["ph_violations"]["stratified"][0] <= M["ph_violations"]["contract_as_covariate"][0]


def test_retention_policy_beats_rules():
    p = M["policy"]
    assert p["survival_policy"]["expected_net_gain"] > p["rule_month_to_month"]["expected_net_gain"] > p["treat_everyone"]["expected_net_gain"]
    assert M["km"]["Month-to-month"]["S(24m)"] < 0.65 and M["km"]["Two year"]["S(24m)"] > 0.99   # contract lock-in


def test_api_high_vs_low_risk_and_curve_is_valid():
    h, l = client.post("/survival", json=HIGH).json(), client.post("/survival", json=LOW).json()
    assert h["survival_probability"]["12m"] < l["survival_probability"]["12m"]
    for r in (h, l):
        cv = r["curve"]
        assert len(cv) == 24 and all(0 <= x <= 1 for x in cv) and all(a >= b - 1e-9 for a, b in zip(cv, cv[1:]))
        assert r["clv_24m_with_offer"] >= r["clv_24m"]
    assert h["recommended_action"] == "offer_retention"


def test_horizon_truncation_at_data_limit_and_validation():
    r = client.post("/survival", json={**HIGH, "tenure": 71}).json()
    assert r["horizon_months_used"] == 1 and r["note"] and r["survival_probability"]["3m"] is None
    assert client.post("/survival", json={**HIGH, "tenure": 72}).json()["recommended_action"] == "no_action"
    assert client.post("/survival", json={**HIGH, "tenure": 0}).status_code == 422
    assert client.post("/survival", json={**HIGH, "tenure": 90}).status_code == 422
