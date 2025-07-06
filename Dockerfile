FROM python:3.8-slim-buster

# Install system dependencies including build tools
RUN apt-get update -y && \
    apt-get install -y \
    awscli \
    build-essential \
    gcc \
    g++ \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core dependencies first
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers datasets evaluate && \
    pip install fastapi uvicorn && \
    pip install pandas PyYAML python-box ensure

# Copy the rest of the application
COPY . /app

# Install the project in development mode
RUN pip install -e .

# Expose port for FastAPI
EXPOSE 8000

CMD ["python3", "app.py"]