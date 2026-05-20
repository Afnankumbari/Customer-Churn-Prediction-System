# 🛡️ ChurnShield — Customer Churn Prediction System

> End-to-end Machine Learning web application for predicting telecom customer churn using Flask and Scikit-learn.

> Python • Flask • Machine Learning • HTML/CSS/JS • Chart.js

---

# 📌 Overview

Customer churn is a major challenge in the telecom industry. Losing customers affects revenue and long-term business growth.

This project predicts **customer churn probability** using Machine Learning and provides:

- Real-time churn prediction
- Interactive dashboard
- Customer data analysis
- Customer database browser
- Filtering & pagination
- Business insights visualization

---

# 🚀 Features

✅ Customer churn prediction

✅ Dashboard with KPIs

✅ Exploratory Data Analysis (EDA)

✅ Customer dataset browser

✅ Filtering & searching

✅ CSV export

✅ Pagination

✅ Flask REST APIs

✅ Interactive charts using Chart.js

---

# 🏗 Architecture

```text
User Input (Frontend)

        ↓

HTML + CSS + JavaScript

        ↓

Flask Backend (app.py)

        ↓

Preprocessing Pipeline

        ↓

Machine Learning Model

        ↓

Prediction Result

        ↓

Dashboard / Charts / Output
```

---

# 📁 Folder Structure

```bash
Customer-Churn-Prediction-System/

│
├── app.py
│
├── templates/
│     ├── base.html
│     ├── dashboard.html
│     ├── predict.html
│     ├── analysis.html
│     └── customers.html
│
├── static/
│     ├── css/
│     │      └── style.css
│     │
│     ├── js/
│     │      ├── dashboard.js
│     │      ├── predict.js
│     │      ├── analysis.js
│     │      └── customers.js
│
├── models/
│     └── best_model.pkl
│
├── data/
│     └── customers.csv
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# 🤖 Machine Learning Pipeline

Dataset

↓

Data Cleaning

↓

Encoding

↓

Feature Scaling

↓

Feature Engineering

↓

Model Training

↓

Evaluation

↓

Best Model Selection

↓

Deployment

---

# 📊 Dataset Information

Dataset contains:

- Customer ID
- Contract Type
- Internet Service
- Monthly Charges
- Total Charges
- Tenure
- Payment Method
- Complaints
- Support Calls
- Churn Status

Records:

```text
5000+ telecom customers
```

---

# 📈 Model Performance

Models tested:

| Model | Purpose |
|------|---------|
| Logistic Regression | Classification |
| Random Forest | Classification |
| Gradient Boosting | Classification |
| SVM | Classification |

Selected model:

```text
Logistic Regression
```

Performance:

```text
Test AUC ≈ 0.775
```

---

# 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | / | Dashboard |
| GET | /predict | Prediction page |
| POST | /api/predict | Predict churn |
| GET | /analysis | EDA page |
| GET | /customers | Customer table |
| GET | /api/customers | Customer JSON |
| GET | /api/stats | Dashboard stats |

---

## Example Request

POST:

```json
{
"Contract":"Month-to-month",
"MonthlyCharges":85,
"Tenure":6,
"InternetService":"Fiber optic"
}
```

Response:

```json
{
"prediction":"High Risk",
"probability":78.2
}
```

---

# ⚙ Technologies Used

Backend:

- Python
- Flask

Machine Learning:

- Scikit-learn
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM

Data Processing:

- Pandas
- NumPy

Visualization:

- Matplotlib
- Seaborn
- Chart.js

Frontend:

- HTML
- CSS
- JavaScript

---

---

# 🚀 Installation

Clone repo:

```bash
git clone https://github.com/Afnankumbari/Customer-Churn-Prediction-System.git
```

Move:

```bash
cd Customer-Churn-Prediction-System
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 🔥 Future Enhancements

- Cloud deployment
- Authentication system
- CRM integration
- Mobile support
- Real-time customer data
- Advanced ML models

---

# 👨‍💻 Author

## Mohammad Afnan Kumbari

Computer Science Engineering

Machine Learning • Data Science • Full Stack Development

GitHub:

https://github.com/Afnankumbari

---

# ⭐ Support

If you found this useful:

Give ⭐ to repository
