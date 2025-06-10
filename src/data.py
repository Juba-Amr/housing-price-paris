import pandas as pd

def load_raw_data(path):
    data = pd.read_csv(path)
    return data 

def clean_data(df):
    y = df.Price
    X = df.drop(columns=['Price','Property Title','Cellar','Beds','Terrace']) 
    X['Living'] = X['Living'].str.replace(' m²', '', regex=False).astype(float) 
    y = y.str.replace(r'[^\d.]', '', regex=True)
    X.to_csv('data/processed/X.csv')
    y.to_csv('data/processed/y.csv')

    return X,y