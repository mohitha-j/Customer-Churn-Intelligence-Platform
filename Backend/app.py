from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.predictor import ChurnPredictor

app = FastAPI(
    title="Customer Churn Intelligence Platform"
)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ChurnPredictor()


class CustomerInput(BaseModel):

    Age: int

    TenureinMonths: int

    MonthlyCharge: float

    Contract: str

    InternetType: str

    PaymentMethod: str

    SatisfactionScore: int

    Number_of_Referrals: int


@app.get("/", response_class=HTMLResponse)
def home():

    with open(
        "templates/index.html",
        encoding="utf-8"
    ) as f:

        return f.read()

@app.get("/dashboard-stats")
def dashboard_stats():

    return {
        "total_customers": 7043,
        "revenue_at_risk": 75954,
        "avg_cltv": 4402
    }
@app.get("/contract-distribution")
def contract_distribution():

    import pandas as pd

    df = pd.read_csv("data/TelcoCustomerChurn.csv")

    data = (
        df["Contract"]
        .value_counts()
        .to_dict()
    )

    return data
@app.get("/churn-segments")
def churn_segments():

    import pandas as pd

    df = pd.read_csv("data/TelcoCustomerChurn.csv")

    churn = (
        df.groupby("Contract")["ChurnLabel"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .round(2)
        .to_dict()
    )

    return churn
@app.get("/customer-status")
def customer_status():

    import pandas as pd

    df = pd.read_csv("data/TelcoCustomerChurn.csv")

    return (
        df["CustomerStatus"]
        .value_counts()
        .to_dict()
    )
@app.get("/model-metrics")
def model_metrics():

    return {
        "accuracy": 0.84,
        "precision": 0.81,
        "recall": 0.79,
        "f1_score": 0.80
    }

@app.post("/predict")
def predict(data: CustomerInput):

    result = predictor.predict(
        data.dict()
    )

    return result