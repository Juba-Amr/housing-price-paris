from sklearn.model_selection import train_test_split, cross_val_score
from data import load_raw_data, clean_data
from pipeline import build_pipe

df = load_raw_data('../data/raw/houses.csv')
X,y = clean_data(df)

X_train, X_val, y_train, y_val = train_test_split(X,y)

pipe = build_pipe(X)

score = cross_val_score(pipe, X_train, y_train,
                        cv=5,
                        scoring='neg_mean_absolute_error')
print(f"model score : {-1*score.mean()}")
