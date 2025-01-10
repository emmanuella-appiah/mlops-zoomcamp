# Use a lightweight Python base image
FROM python:3.11.0-slim

# Upgrade pip and install pipenv
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pipenv

# Install MLflow
RUN pip install --no-cache-dir mlflow[extras]

# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock to the container
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies globally (system-wide)
RUN pipenv install --system --deploy

# Copy models from local directory to the container
COPY ./models /app/models

# Copy your application code (Flask app)
COPY deployment/predict.py .

# Expose the ports used by Flask and MLflow
EXPOSE 9696 5000

# Command to run MLflow server and Flask app concurrently
CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000 & python predict.py"]
