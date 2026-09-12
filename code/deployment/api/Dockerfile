FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY requirements-api.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir -r requirements-api.txt
COPY code ./code
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "code.deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
