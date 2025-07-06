FROM python:3.8-slim-buster

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . /app

# Install the project in development mode
RUN pip install -e .

# Expose port for FastAPI
EXPOSE 8000

CMD ["python3", "app.py"]