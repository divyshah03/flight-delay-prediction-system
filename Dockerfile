FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY evaluation/ evaluation/
COPY models/ models/
COPY features/ features/
COPY data/config.py data/airports.py data/__init__.py data/
COPY artifacts/ artifacts/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
