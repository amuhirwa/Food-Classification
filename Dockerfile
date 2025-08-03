FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (like libGL)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy only requirements.txt to cache dependencies
COPY requirements.txt .

# Install Python packages (only re-runs if requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy app code — changes here won't affect the pip install cache
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY ui/ ./ui/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p uploads retrain_data logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV UPLOAD_DIR=/app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
