# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies ONCE during build
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Default command to run the server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]