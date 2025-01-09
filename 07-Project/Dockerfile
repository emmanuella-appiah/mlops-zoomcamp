# Use a lightweight Python base image
FROM python:3.11.0-slim

# Upgrade pip and install pipenv
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pipenv

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

# Expose the port used by the Flask app
EXPOSE 9696

# Command to run your Flask app directly
CMD ["python", "predict.py"]
