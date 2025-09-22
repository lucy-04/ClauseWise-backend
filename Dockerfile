# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with no cache for smaller image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ .


# Create necessary directories
RUN mkdir -p chroma_db

# Expose the port Cloud Run will use
EXPOSE 8080

# Use the PORT environment variable from Cloud Run
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 30