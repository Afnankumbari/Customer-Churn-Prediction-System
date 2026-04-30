-- ============================================================
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
