"""
generate_data.py
----------------
Generates a realistic telecom customer churn dataset (5000 rows).
Outputs: data/customers.csv + data/schema.sql + data/inserts.sql
"""

import numpy as np
import pandas as pd
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N = 5000

def generate_dataset(n: int = N) -> pd.DataFrame:
    customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, n + 1)]
    tenure = np.random.randint(1, 73, n)                         # months
    age = np.random.randint(18, 75, n)
    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"],
                                n, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == "No", "No phone service",
                              np.random.choice(["Yes", "No"], n))
    online_security = np.where(internet_service == "No", "No internet service",
                               np.random.choice(["Yes", "No"], n))
    online_backup = np.where(internet_service == "No", "No internet service",
                             np.random.choice(["Yes", "No"], n))
    device_protection = np.where(internet_service == "No", "No internet service",
                                 np.random.choice(["Yes", "No"], n))
    tech_support = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], n))
    streaming_tv = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], n))
    streaming_movies = np.where(internet_service == "No", "No internet service",
                                np.random.choice(["Yes", "No"], n))
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    num_complaints = np.random.poisson(0.5, n)
    support_calls = np.random.poisson(1.5, n)

    # Churn probability driven by key factors
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.10 * (internet_service == "Fiber optic")
        + 0.08 * (payment_method == "Electronic check")
        + 0.05 * (paperless_billing == "Yes")
        - 0.15 * (tenure > 36)
        - 0.10 * (contract == "Two year")
        + 0.04 * (senior_citizen == 1)
        + 0.03 * (num_complaints > 1)
        + 0.02 * (support_calls > 3)
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    # Introduce ~2% missing values in MonthlyCharges & TotalCharges
    for col_arr in [monthly_charges, total_charges]:
        miss_idx = np.random.choice(n, int(0.02 * n), replace=False)
        col_arr[miss_idx] = np.nan  # will be handled in preprocessing

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "Age": age,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "NumComplaints": num_complaints,
        "SupportCalls": support_calls,
        "Churn": churn,
    })
    return df


def write_sql(df: pd.DataFrame, out_dir: str):
    schema = """-- ============================================================
-- Customer Churn Prediction Schema
-- Compatible with: SQLite / MySQL
-- ============================================================

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    CustomerID      VARCHAR(12) PRIMARY KEY,
    Gender          VARCHAR(10),
    SeniorCitizen   TINYINT DEFAULT 0,
    Partner         VARCHAR(5),
    Dependents      VARCHAR(5),
    Tenure          INT,
    Age             INT,
    PhoneService    VARCHAR(5),
    MultipleLines   VARCHAR(20),
    InternetService VARCHAR(15),
    OnlineSecurity  VARCHAR(25),
    OnlineBackup    VARCHAR(25),
    DeviceProtection VARCHAR(25),
    TechSupport     VARCHAR(25),
    StreamingTV     VARCHAR(25),
    StreamingMovies VARCHAR(25),
    Contract        VARCHAR(20),
    PaperlessBilling VARCHAR(5),
    PaymentMethod   VARCHAR(25),
    MonthlyCharges  DECIMAL(8,2),
    TotalCharges    DECIMAL(10,2),
    NumComplaints   INT DEFAULT 0,
    SupportCalls    INT DEFAULT 0,
    Churn           TINYINT DEFAULT 0
);

-- Useful analytical queries
-- 1. Churn rate by contract type
-- SELECT Contract, COUNT(*) AS Total,
--        SUM(Churn) AS Churned,
--        ROUND(100.0*SUM(Churn)/COUNT(*),2) AS ChurnRate
-- FROM customers GROUP BY Contract ORDER BY ChurnRate DESC;

-- 2. Avg monthly charges for churned vs retained
-- SELECT Churn, ROUND(AVG(MonthlyCharges),2) AS AvgMonthly
-- FROM customers GROUP BY Churn;

-- 3. High-risk customers
-- SELECT CustomerID, Tenure, MonthlyCharges, Contract
-- FROM customers WHERE Churn=1 AND Tenure < 12
-- ORDER BY MonthlyCharges DESC LIMIT 50;
"""
    with open(os.path.join(out_dir, "schema.sql"), "w") as f:
        f.write(schema)

    # Write batch inserts (500 rows per batch for efficiency)
    lines = ["-- Auto-generated INSERT statements\n"]
    batch_size = 500
    cols = ",".join(df.columns)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        values_list = []
        for _, row in chunk.iterrows():
            vals = []
            for v in row:
                if pd.isna(v):
                    vals.append("NULL")
                elif isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    vals.append(str(round(float(v), 2)))
                else:
                    escaped = str(v).replace("'", "''")
                    vals.append(f"'{escaped}'")
            values_list.append(f"({','.join(vals)})")
        lines.append(f"INSERT INTO customers ({cols}) VALUES\n" +
                     ",\n".join(values_list) + ";\n")

    with open(os.path.join(out_dir, "inserts.sql"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_dataset()
    df.to_csv(os.path.join(out_dir, "customers.csv"), index=False)
    write_sql(df, out_dir)
    print(f"✅ Dataset saved: {len(df)} rows | Churn rate: {df['Churn'].mean():.1%}")
    print(f"   Files: data/customers.csv, data/schema.sql, data/inserts.sql")
