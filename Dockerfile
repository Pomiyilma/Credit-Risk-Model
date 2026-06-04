FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files to disk and ensure direct console stream piping
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy over local directories
COPY ./src /app/src
COPY ./mlruns /app/mlruns

EXPOSE 8000

# 👇 THE UPDATED FIX LOGIC 👇
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
