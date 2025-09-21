FROM python:3.11-slim

WORKDIR /app

# Copy requirements from root
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code into container
COPY backend/ ./backend

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (main.py is inside /app/backend/)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
