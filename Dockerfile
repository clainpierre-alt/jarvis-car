FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Cloud Run health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Expose port
EXPOSE 8080

# Run
CMD exec gunicorn main:app \
    --bind :${PORT:-8080} \
    --workers 1 \
    --threads 2 \
    --timeout 0 \
    --log-level info
