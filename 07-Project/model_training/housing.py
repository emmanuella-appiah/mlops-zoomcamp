#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
import joblib
import os


# In[2]:


from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['Price'] = california_housing.target

df.head(3)


# In[3]:


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


# In[4]:


df.columns, df.dtypes, df.shape


# In[5]:


df = df.astype({
    'HouseAgeYears': 'int64',
    'TotalPopulation': 'int64',
})


# In[6]:


df_corr = df.corr()
plt.figure(figsize=(10, 8))  
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')


# In[7]:


X = df.drop(columns="HousePrice", axis=1)
y = df['HousePrice']


# In[8]:


# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, X_valid shape: {X_valid.shape}, X_test shape: {X_test.shape}")


# In[10]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns)
    ])


# In[11]:


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])


# In[12]:


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("project-7") 


# In[13]:


with mlflow.start_run():
    mlflow.set_tag("developer","Emmanuella")
    
    params = {"random_state": 42}
    mlflow.log_params(params)
    
    # model = RandomForestRegressor(**params)
    # model.fit(X_train, y_train)
    
    # y_pred = model.predict(X_test)
    model_pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = model_pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    mlflow.log_metric("Mean Squared Error", mse)
    
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    mlflow.log_metric("Root Mean Squared Error", rmse)
    
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score (Accuracy): {r2}")
    mlflow.log_metric("R2 Score", r2)
    
    mlflow.sklearn.log_model(model_pipeline, artifact_path="../models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

    os.makedirs("../models", exist_ok=True)
    model_path = "../models/lin_reg.bin"
    joblib.dump(model_pipeline, model_path)
    mlflow.log_artifact(local_path="../models/lin_reg.bin", artifact_path="models_pickle")


# ## More Model Experiments

# In[14]:


import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


# In[15]:


train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_valid, label=y_valid)


# In[16]:


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse) 
    return {'loss': rmse, 'accuracy(r2 score)': r2, 'status': STATUS_OK}


# In[17]:


search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}


# In[18]:


best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)


# In[19]:


import pickle
with mlflow.start_run():
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_valid, label=y_valid)

    best_params ={
    "learning_rate" : 0.07068105887408552,
    "max_depth" : 5,
    "min_child_weight": 0.5129258540979367,
    "objective" : "reg:linear",
    "reg_alpha" : 0.1959651381094002,
    "reg_lambda": 0.008613854781972727,
    "seed" : 42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


# In[24]:


import mlflow
logged_model = 'runs:/a82512716a004af69ff176a4714fe426/models_mlflow'
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model


# In[25]:


xgboost_model = mlflow.xgboost.load_model(logged_model)
y_pred = xgboost_model.predict(valid)


# In[26]:


r2 = r2_score(y_valid, y_pred)


# In[27]:


r2


# In[ ]:




