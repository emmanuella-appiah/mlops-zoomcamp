#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os


def read_data(filename):
    return pd.read_parquet(filename)


def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def get_input_path(year, month):
    default_input_pattern = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-' \
                            f'{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern


def get_output_path(year, month):
    default_output_pattern = f's3://nyc-duration-emmanuella/taxi_type=fhv/year={year:04d}/month' \
                             f'={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern


def main(month, year):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ['PULocationID', 'DOLocationID']

    data = read_data(input_file)
    df = prepare_data(df=data, categorical=categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')

    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print('Predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred,
    })
    print(f"Output file path: {output_file}")
    storage_options = {
        'client_kwargs': {
            'endpoint_url': os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
        }
    }
    df_result.to_parquet(output_file, engine='pyarrow', storage_options=storage_options, index=False)


if __name__ == "__main__":
    month = int(sys.argv[1])
    year = int(sys.argv[2])
    main(month, year)


# python batch.py 3 2023
#  pytest tests/test_batch.py


# s3://nyc-duration-emmanuella/taxi_type=fhv/year=2023/month=03/predictions.parquet
