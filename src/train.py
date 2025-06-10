from sklearn.model_selection import cross_val_score
from data import load_raw_data, clean_data
from pipeline import build_pipe
import joblib
import os


df = load_raw_data('data/raw/houses.csv')
X,y = clean_data(df)


pipe = build_pipe(X)

score = cross_val_score(pipe, X, y,
                        cv=5,
                        scoring='neg_mean_absolute_error')
print(f"model score : {-1*score.mean()}")

pipe.fit(X,y)

os.makedirs('model', exist_ok=True)
joblib.dump(pipe, 'model/house_price_pipeline.joblib')


