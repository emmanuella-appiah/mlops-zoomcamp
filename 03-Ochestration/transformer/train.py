if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


@transformer
def transform(df, *args, **kwargs):
    dv = DictVectorizer()

    categorical = df[['PULocationID', 'DOLocationID']].astype('str')
    cat_dict = categorical.to_dict(orient="records")

    X_train = dv.fit_transform(cat_dict)
    y_train = df.duration.values

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(model.intercept_)
    return dv, model