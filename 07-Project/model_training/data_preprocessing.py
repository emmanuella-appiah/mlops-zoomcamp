import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from prefect import task


@task
def load_data():
    california_housing = fetch_california_housing()
    df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    df['HousePrice'] = california_housing.target

    df.rename(columns={
        'MedInc': 'MedianIncome',
        'HouseAge': 'HouseAgeYears',
        'AveRooms': 'AverageRoomsPerHouse',
        'AveBedrms': 'AverageBedroomsPerHouse',
        'Population': 'TotalPopulation',
        'AveOccup': 'AverageOccupantsPerHouse',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'Price': 'HousePrice'
    }, inplace=True)

    df = df.astype({'HouseAgeYears': 'int64', 'TotalPopulation': 'int64'})

    return df


@task
def preprocess_data(df):
    X = df.drop(columns="HousePrice", axis=1)
    y = df['HousePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns)
        ])

    return X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor
