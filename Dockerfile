# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app


# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get install -y build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port (default 8000, can be changed via config)
EXPOSE 8000

# Use uvicorn for production serverless deployment
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
