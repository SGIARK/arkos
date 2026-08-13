FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 1112

# harness_module/api.py is written in Task 4. Until then this image builds but
# does not serve.
CMD ["python", "-m", "uvicorn", "harness_module.api:app", "--host", "0.0.0.0", "--port", "1112"]
