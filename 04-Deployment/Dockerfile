FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

RUN pip install -U pip && pip install pipenv

COPY Pipfile Pipfile.lock starter.py model.bin ./

RUN pipenv install --system --deploy

RUN mkdir /app/output

COPY data/yellow_tripdata_2023-03.parquet /app/data/yellow_tripdata_2023-03.parquet
COPY data/yellow_tripdata_2023-04.parquet /app/data/yellow_tripdata_2023-04.parquet
COPY data/yellow_tripdata_2023-05.parquet /app/data/yellow_tripdata_2023-05.parquet

ENTRYPOINT [ "python", "starter.py" ]

