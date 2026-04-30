# 🛡️ ChurnShield — Python + Flask + HTML/CSS

> **Production-ready** Customer Churn Prediction  
> Python backend · Flask API · Sklearn ML · Beautiful HTML/CSS frontend

---

## 🏗️ Architecture

```
Browser (HTML/CSS/JS)
        ↕  HTTP
Flask Server (app.py)
        ↕  Python
sklearn Pipeline (models/best_model.pkl)
        ↕  Pandas
customers.csv / SQLite DB
```

## 📁 Folder Structure

```
churnshield_flask/
├── app.py                  ← Flask server (routes + API endpoints)
├── run_pipeline.py         ← One-shot: generate data → EDA → train model
├── requirements.txt
├── README.md
├── Dockerfile
│
├── src/
│   ├── generate_data.py    ← Creates 5,000-row telecom dataset
│   ├── preprocess.py       ← Feature engineering + sklearn pipelines
│   ├── eda.py              ← Generates matplotlib plots
│   └── train.py            ← Trains & compares 4 ML models
│
├── templates/              ← Jinja2 HTML templates (Flask)
│   ├── base.html           ← Nav, footer, shared layout
│   ├── index.html          ← Dashboard homepage
│   ├── predict.html        ← Prediction form + results
│   ├── eda.html            ← EDA charts + matplotlib plots
│   ├── models.html         ← Model comparison page
│   ├── customers.html      ← Filterable data table
│   └── setup.html          ← Shown if pipeline not run yet
│
├── static/
│   ├── css/main.css        ← All styles (no framework)
│   └── js/main.js          ← Chart.js + fetch API calls
│
├── data/
│   ├── customers.csv
│   ├── schema.sql
│   └── inserts.sql
│
├── models/
│   └── best_model.pkl      ← sklearn Pipeline (pickle)
│
└── assets/                 ← 10 matplotlib PNG plots
```

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data, run EDA, train models
python run_pipeline.py

# 3. Start Flask server
python app.py

# 4. Open browser
#    http://127.0.0.1:5000
```

## 🌐 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Dashboard homepage |
| GET | `/predict` | Prediction form |
| **POST** | `/api/predict` | **JSON prediction endpoint** |
| GET | `/eda` | EDA analysis page |
| GET | `/models` | Model performance page |
| GET | `/customers` | Customer data table |
| GET | `/api/customers` | JSON customer data (filterable) |
| GET | `/api/stats` | JSON dataset statistics |

### POST /api/predict — Example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "Tenure": 6,
    "MonthlyCharges": 85,
    "TotalCharges": 510,
    "PaymentMethod": "Electronic check",
    "SeniorCitizen": 0,
    "NumComplaints": 1,
    "SupportCalls": 2
  }'
```

**Response:**
```json
{
  "probability": 0.7823,
  "probability_pct": 78.2,
  "prediction": 1,
  "risk_level": "HIGH",
  "model_name": "Logistic Regression",
  "model_auc": 0.775,
  "risk_factors": [...]
}
```

## 🚀 Deployment

### Local
```bash
python app.py
```

### Docker
```bash
docker build -t churnshield .
docker run -p 5000:5000 churnshield
# → http://localhost:5000
```

### AWS EC2
```bash
pip install -r requirements.txt
python run_pipeline.py
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Render / Railway (Free)
- Push to GitHub
- Set build command: `pip install -r requirements.txt && python run_pipeline.py`
- Set start command: `python app.py`

## 📄 License
MIT © 2024 ChurnShield
