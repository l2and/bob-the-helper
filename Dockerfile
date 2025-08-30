# Use Python 3.11 slim image for better performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=main.py \
    FLASK_ENV=production \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY happyLittleTreesOfKnowledge/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir debugpy

# Copy application code
COPY happyLittleTreesOfKnowledge/ .

# Create non-root user for security (skip in debug mode)
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
# USER app - commented out for debugging compatibility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports (app and debugger)
EXPOSE ${PORT}
EXPOSE 5678

# Default command - can be overridden for debugging
CMD ["sh", "-c", "python -m debugpy --wait-for-client --listen 0.0.0.0:5678 main.py || python main.py"]