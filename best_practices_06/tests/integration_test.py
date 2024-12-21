import pandas as pd
from datetime import datetime
import os


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def save_to_s3(df, output_file, s3_endpoint):
    options = {'client_kwargs': {'endpoint_url': s3_endpoint}}
    df.to_parquet(output_file, engine='pyarrow', compression=None, index=False, storage_options=options)
    print(f"DataFrame saved successfully to {output_file}.")


def load_from_s3(output_file, s3_endpoint):
    options = {'client_kwargs': {'endpoint_url': s3_endpoint}}
    return pd.read_parquet(output_file, storage_options=options)


def get_sum_of_predicted_durations(output_file):
    df_result = pd.read_parquet(output_file, storage_options={'client_kwargs': {'endpoint_url': os.getenv('S3_ENDPOINT_URL')}})
    return df_result['predicted_duration'].sum()


def test_save_to_s3():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df['predicted_duration'] = [9.0, 8.0, 7.0, 10.0]

    input_file = 's3://nyc-duration-emmanuella/taxi_type=fhv/year=2023/month=01/integration_test.parquet'
    s3_endpoint = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')

    save_to_s3(df, input_file, s3_endpoint)
    os.system(f"aws --endpoint-url={s3_endpoint} s3 ls {input_file}")

    df_loaded = load_from_s3(input_file, s3_endpoint)
    print(df_loaded.head(2))
    print("Sum of predicted durations:", df_loaded['predicted_duration'].sum())


if __name__ == "__main__":
    test_save_to_s3()
