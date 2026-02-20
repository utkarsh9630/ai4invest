# AI4Invest - Financial Risk Profiling Platform

## Overview

AI4Invest is an intelligent financial risk profiling platform that uses machine learning to assess users' risk tolerance and provide personalized S&P 500 stock recommendations. The system analyzes 19 financial and demographic factors to classify users into risk categories and suggests tailored investment strategies.

**Tagline:** *Why retire so late? Join us and retire early with smart investments!*

## Key Features

-  **AI-Powered Risk Assessment** - ML-driven analysis of financial profiles
-  **Personalized Stock Recommendations** - Top 5 S&P 500 picks based on risk profile
-  **Investment Simulator** - Test investment scenarios with predicted returns
-  **Profile Management** - Save and compare multiple financial profiles
-  **Secure Authentication** - User registration with encrypted password storage
-  **Responsive Design** - Works seamlessly on desktop and mobile devices

## Technology Stack

**Backend:**
- Flask 3.1.2 - Web framework
- PostgreSQL - Production database
- SQLAlchemy - ORM and database migrations
- Gunicorn - WSGI server

**Machine Learning:**
- scikit-learn 1.6.1 - Risk classification & return prediction
- pandas 2.2.2 - Data processing
- NumPy 1.26.4 - Numerical computations

**Frontend:**
- Tailwind CSS - Modern UI styling
- Vanilla JavaScript - Interactive features

## Project Structure
```
ai4invest/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── base.html                  # Base layout
│   ├── login.html                 # Login page
│   ├── register.html              # Registration page
│   ├── form.html                  # Risk profiling form (paginated)
│   ├── dashboard.html             # Stock recommendations dashboard
│   ├── profiles.html              # Saved profiles management
│   └── simulation.html            # Investment simulator
├── static/
│   └── globals.css                # Custom styling
├── migrations/                     # Database migrations
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version
├── Procfile                       # Render deployment config
├── Dockerfile                     # Docker configuration
└── ML Models & Data:
    ├── risk_pipeline.joblib       # Risk classification model
    ├── risk_label_encoder.joblib  # Label encoder
    ├── stock_classifier.joblib    # Stock clustering model
    ├── topreturn_model.joblib     # Return prediction model
    ├── top_n_per_category.csv     # Stock recommendations
    └── sp500_features.csv         # S&P 500 features dataset
```

## Machine Learning Pipeline

The application uses three integrated ML models:

1. **Risk Profiler** (`risk_pipeline.joblib`) - Classifies users into Low/Medium/High risk categories based on 19 financial indicators
2. **Stock Classifier** (`stock_classifier.joblib`) - K-means clustering of S&P 500 stocks by risk profile
3. **Return Predictor** (`topreturn_model.joblib`) - Random Forest model for 90-day return forecasts

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

## Deployment

The application is deployed on Render.com with:
- **Web Service:** Python 3.12, Gunicorn
- **Database:** PostgreSQL (free tier - 90 days)
- **Auto-deploy:** Enabled from main branch

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

MIT License - Educational purposes

## Contact

**Author:** Utkarsh Tripathi  
**GitHub:** [@utkarsh9630](https://github.com/utkarsh9630)

---

⚠️ **Disclaimer:** This is a demonstration application for educational purposes. Always consult with a qualified financial advisor before making investment decisions.
