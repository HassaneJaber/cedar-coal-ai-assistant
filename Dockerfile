FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/app
RUN useradd -m -u 1000 app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app . .
RUN mkdir -p data .files && chown app:app /app data .files
USER app
EXPOSE 8000
CMD ["sh", "-c", "exec chainlit run app.py --host 0.0.0.0 --port ${PORT:-8000} --headless"]
