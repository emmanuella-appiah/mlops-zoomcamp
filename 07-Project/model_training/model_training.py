import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error
from xgboost import DMatrix, train
from sklearn.pipeline import Pipeline
from utils import log_artifacts
from prefect import task


@task
def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    with mlflow.start_run():
        mlflow.set_tag("developer", "Emmanuella")
        model_pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = model_pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("R2 Score", r2)

        # Log the model
        mlflow.sklearn.log_model(model_pipeline, artifact_path="ramdom_forest_models")
        log_artifacts(model_pipeline)


@task
def train_xgboost(X_train, y_train, X_valid, y_valid):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10),
            "objective": "reg:squarederror",
            "seed": 42
        }

        train_data = DMatrix(X_train, label=y_train)
        valid_data = DMatrix(X_valid, label=y_valid)

        booster = train(
            params=params,
            dtrain=train_data,
            num_boost_round=1000,
            evals=[(valid_data, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        y_pred = booster.predict(valid_data)
        mse = mean_squared_error(y_valid, y_pred)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    train_data = DMatrix(X_train, label=y_train)
    valid_data = DMatrix(X_valid, label=y_valid)

    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(best_params)

        booster = train(
            params=best_params,
            dtrain=train_data,
            num_boost_round=1000,
            evals=[(valid_data, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid_data)

        r2 = r2_score(y_valid, y_pred)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = mse ** 0.5

        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)
        mlflow.log_metric("R2 Score", r2)

        mlflow.xgboost.log_model(booster, artifact_path="xboost_models_mlflow")
        print(f"Model logged with R2 Score: {r2}")
