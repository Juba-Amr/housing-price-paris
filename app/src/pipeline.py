import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

def build_pipe(X):
    enc = OneHotEncoder(handle_unknown='ignore')

    cat_cols = [col for col in X.columns if X[col].dtype == 'object']
    num_cols = [col for col in X.columns if col not in cat_cols]


    num_transformer = SimpleImputer(strategy='mean')
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', enc)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    pipe = Pipeline([
        ('preprocess',preprocessor),
        ('model',XGBRegressor(n_estimators=500, learning_rate=0.1, n_jobs=4, seed=4))
    ])
    return pipe