# AI4Invest — Financial Risk Profiling Platform

## Overview

AI4Invest is an intelligent financial risk profiling platform that uses machine learning to assess users' risk tolerance and provide personalized S&P 500 stock recommendations. The system analyzes 19 financial and demographic factors to classify users into risk categories and suggests tailored investment strategies.

**Tagline:** *Why retire so late? Join us and retire early with smart investments!*

## Key Features

- **AI-Powered Risk Assessment** — ML-driven analysis of financial profiles
- **Personalized Stock Recommendations** — Top 5 S&P 500 picks based on risk profile
- **Investment Simulator** — Test investment scenarios with predicted returns
- **Profile Management** — Save and compare multiple financial profiles
- **Secure Authentication** — User registration with encrypted password storage
- **Responsive Design** — Works seamlessly on desktop and mobile devices

## Technology Stack

**Backend:**
- Flask 3.1.0 — Web framework
- SQLAlchemy — ORM and database migrations
- Werkzeug 3.1.3 — WSGI utilities and security

**Machine Learning:**
- scikit-learn 1.6.1 — Risk classification & return prediction
- pandas 2.2.2 — Data processing
- NumPy 1.26.4 — Numerical computations
- joblib 1.3.2 — Model persistence

**Frontend:**
- Tailwind CSS — Modern UI styling
- Vanilla JavaScript — Interactive features

**Deployment:**
- Vercel — Serverless hosting via Python WSGI

## Project Structure

```
ai4invest/
├── api/
│   └── index.py                    # Vercel entry point — exposes Flask app
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── form.html                   # Risk profiling form (paginated)
│   ├── dashboard.html              # Stock recommendations dashboard
│   ├── profiles.html               # Saved profiles management
│   └── simulation.html             # Investment simulator
├── static/
│   └── globals.css                 # Custom styling
├── migrations/                     # Database migrations
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel routing configuration
└── ML Models & Data:
    ├── risk_pipeline.joblib        # Risk classification model
    ├── risk_label_encoder.joblib   # Label encoder
    ├── topreturn_model.joblib      # Return prediction model
    ├── top_n_per_category.csv      # Stock recommendations
    └── sp500_features.csv          # S&P 500 features dataset
```

## Machine Learning Pipeline

The application uses three integrated ML models:

1. **Risk Profiler** (`risk_pipeline.joblib`) — Classifies users into Low/Medium/High risk categories based on 19 financial indicators
2. **Stock Classifier** (`stock_classifier.joblib`) — K-means clustering of S&P 500 stocks by risk profile
3. **Return Predictor** (`topreturn_model.joblib`) — Random Forest model for 90-day return forecasts

## Local Development

```bash
# Clone repository
git clone https://github.com/utkarsh9630/ai4invest.git
cd ai4invest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Access at http://localhost:5000
```

---

## Deployment to Vercel

### What was changed for Vercel compatibility

| File | Change | Reason |
|---|---|---|
| `vercel.json` | Added — routes all requests to `api/index.py` | Vercel needs explicit routing config for Python apps |
| `api/index.py` | Added — exports the Flask `app` object | Vercel's WSGI runtime requires an `app` object in `api/` |
| `app.py` | All `joblib.load()` / `pd.read_csv()` calls changed to use `Path(__file__).parent` | Vercel's serverless working directory is not guaranteed to be the project root — bare filenames like `joblib.load("risk_pipeline.joblib")` raise `FileNotFoundError` |
| `app.py` | SQLite path changed to `/tmp/app.db` for non-Render environments | `/tmp` is the only writable directory on Vercel |
| `requirements.txt` | Removed `gunicorn`, `python-docx===`, `python-3.11.9`; pinned all Flask versions | `gunicorn` is unneeded on Vercel; `python-docx===` and `python-3.11.9` are invalid pip packages that crash the build |

### Deploy steps

1. **Push to GitHub:**
```bash
git add .
git commit -m "Vercel deployment"
git push origin main
```

2. **Import to Vercel:**
- Go to [vercel.com](https://vercel.com) → **Add New Project**
- Import your GitHub repository

3. **Add environment variables** in Vercel → Settings → Environment Variables:

| Key | Value |
|---|---|
| `SECRET_KEY` | A long random string — run `python -c "import secrets; print(secrets.token_hex(32))"` |

4. **Deploy** — Vercel builds and publishes automatically. Redeploys on every `git push`.

> **Note on database persistence:** The default SQLite in `/tmp` is ephemeral — saved profiles are lost between cold starts. For persistent user profiles, set `DATABASE_URL` to a free-tier PostgreSQL instance from [Supabase](https://supabase.com) or [Neon](https://neon.tech).

---

## Features in Detail

### Risk Profiling Form
- 19-question assessment across 5 pages
- Categories: Demographics, Financial Situation, Outlook, Investments, Self-Assessment
- Progress tracking and validation

### Dashboard
- Risk classification result (Low/Medium/High)
- Top 5 personalized stock recommendations
- Predicted returns and risk indicators
- CSV export functionality

### Investment Simulator
- Test different investment amounts
- Adjust time horizons (days)
- View projected gains/losses
- Compare scenarios

## Contributing

This is an academic project. Feedback and suggestions are welcome!

## License

MIT License — Educational purposes

## Contact

**Utkarsh Tripathi**
- GitHub: [@utkarsh9630](https://github.com/utkarsh9630)
- LinkedIn: [Utkarsh Tripathi](https://www.linkedin.com/in/tripathiutkarsh46/)

---

⚠️ **Disclaimer:** This is a demonstration application for educational purposes. Always consult with a qualified financial advisor before making investment decisions.
