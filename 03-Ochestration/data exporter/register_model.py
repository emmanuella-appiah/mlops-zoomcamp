if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
    
import mlflow
import pickle

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment('homework-03')


@data_exporter
def export_data(df, *args, **kwargs):
    dv, model = df
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear-regression-model")
        with open('dict_vectorizer.pkl', 'wb') as f:
            pickle.dump(dv, f)

        mlflow.log_artifact('dict_vectorizer.pkl')
