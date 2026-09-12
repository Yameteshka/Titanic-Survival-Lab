FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY code/deployment/app/app.py ./app.py
COPY .streamlit ./.streamlit
EXPOSE 8501
HEALTHCHECK --interval=20s --timeout=5s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
