import mlflow
import joblib
import os
from prefect import task

@task
def log_artifacts(model_pipeline):
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/lin_reg.pkl"
    joblib.dump(model_pipeline, model_path)
    mlflow.log_artifact(local_path=model_path, artifact_path="models_pickle")
