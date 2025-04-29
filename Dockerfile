FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scikit-learn joblib

# Copy all files
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py"] 