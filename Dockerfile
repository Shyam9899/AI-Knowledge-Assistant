FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]