import mlflow
from data_preprocessing import load_data, preprocess_data
from model_training import train_random_forest, train_xgboost
from utils import log_artifacts
from prefect import task, flow  #


@task
def load_and_preprocess():
    df = load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor = preprocess_data(df)
    return X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor


@task
def train_rf(X_train, y_train, X_test, y_test, preprocessor):
    train_random_forest(X_train, y_train, X_test, y_test, preprocessor)


@task
def train_xgb(X_train, y_train, X_valid, y_valid):
    train_xgboost(X_train, y_train, X_valid, y_valid)


@flow(name="MLFlow-Prefect-Workflow")
def run_workflow():
    X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor = load_and_preprocess()

    train_rf(X_train, y_train, X_test, y_test, preprocessor)
    train_xgb(X_train, y_train, X_valid, y_valid)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("project-7")
    run_workflow()
