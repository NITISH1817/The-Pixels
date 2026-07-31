# ============================================================
# Dockerfile - Student Performance MLOps Prediction App
# ============================================================

# 1. Use official lightweight Python 3.13 image
FROM python:3.13-slim

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# 3. Set working directory in container
WORKDIR /app

# 4. Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy source code, model artifacts, monitoring, and API frontend files
COPY src/ ./src/
COPY models/ ./models/
COPY api/ ./api/
COPY data/ ./data/
COPY monitoring/ ./monitoring/

# 7. Expose port 8000
EXPOSE 8000

# 8. Command to launch FastAPI prediction app on container start
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
