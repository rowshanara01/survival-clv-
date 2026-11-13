"""Survival API: 'kobe' chhere jabe + survival-based CLV + offer dile laabh kina"""
import json
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import common as C

B = joblib.load("model/survival_model.joblib")
CPH, PREP = B["cph"], B["prep"]
app = FastAPI(title="Customer Survival & CLV API")
YN = Literal["Yes", "No"]
ADDON = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    tenure: int = Field(ge=1, le=72, description="koto mash dhore (data 72 mash porjonto-i ache)")
    monthly_charges: float = Field(gt=0, le=500)
    senior_citizen: Literal[0, 1] = 0
    partner: YN
    dependents: YN
    phone_service: YN
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: ADDON
    online_backup: ADDON
    device_protection: ADDON
    tech_support: ADDON
    streaming_tv: ADDON
    streaming_movies: ADDON
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YN
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


MAP = {"monthly_charges": "MonthlyCharges", "senior_citizen": "SeniorCitizen", "partner": "Partner",
       "dependents": "Dependents", "phone_service": "PhoneService", "multiple_lines": "MultipleLines",
       "internet_service": "InternetService", "online_security": "OnlineSecurity", "online_backup": "OnlineBackup",
       "device_protection": "DeviceProtection", "tech_support": "TechSupport", "streaming_tv": "StreamingTV",
       "streaming_movies": "StreamingMovies", "contract": "Contract", "paperless_billing": "PaperlessBilling",
       "payment_method": "PaymentMethod", "tenure": "tenure"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return json.load(open("model/metrics.json"))


@app.post("/survival")
def survival(c: Customer):
    row = pd.DataFrame([{MAP[k]: v for k, v in c.model_dump().items()}])
    S = C.surv(CPH, PREP.transform(row), np.arange(0, C.MAX_T + 1))[0]
    cond = C.conditional_curve(S, c.tenure, C.ECON["horizon"])   # ekhon thekhe porer prottek mash e thakar probability
    value, k = C.clv(S, c.tenure, c.monthly_charges)
    treated, _ = C.clv(S, c.tenure, c.monthly_charges, C.ECON["hazard_multiplier"])
    gain = treated - value - C.ECON["offer_cost"]
    at = lambda m: round(float(cond[m - 1]), 4) if len(cond) >= m else None
    return {"survival_probability": {"3m": at(3), "6m": at(6), "12m": at(12), "24m": at(24)},
            "expected_months_retained_next_24": round(float(cond.sum()), 2),
            "clv_24m": round(value, 2), "clv_24m_with_offer": round(treated, 2),
            "offer_net_gain": round(gain, 2),
            "recommended_action": "offer_retention" if gain > 0 else "no_action",
            "horizon_months_used": k,
            "note": None if k == C.ECON["horizon"] else f"Data 72 mash porjonto-i ache, tai shudhu porer {k} mash dhora hoyeche",
            "curve": [round(float(x), 4) for x in cond]}
