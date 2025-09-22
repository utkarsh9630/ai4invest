# Force Python 3.11 so numpy/scipy/sklearn wheels are available
FROM python:3.11-slim

# Install OS deps only if you later need them (commented now)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Start with Gunicorn; Render injects $PORT, so bind to it
CMD gunicorn -b 0.0.0.0:$PORT app:app
