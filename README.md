This is my first machine learning project
This model's objective is predicting the selling price of houses in Paris using a database available on Kaggle
We will be comparing RandomForest from scikit-learn and XGBoost for optimal predictions
And this is mainly to show what I learned on Kaggle's courses

Houses Selling Price Prediction - Streamlit App
Description
This project is an interactive web app built with Streamlit that predicts housing sale prices in Paris.
The goal is to provide quick and reliable price estimates based on real data, using a complete machine learning pipeline.

Features
Loading and cleaning raw data

Building a preprocessing and modeling pipeline with scikit-learn (version 1.7)

Predicting sale prices from property features

Simple, intuitive user interface for easy model testing

Handling versioning and error issues (retrained with sklearn 1.7)

Technologies Used
Python 3.12

pandas, matplotlib

scikit-learn (1.7)

xgboost

joblib (for model serialization)

Streamlit (for interactive web deployment)

How to Run
Clone the repo

Install dependencies:


>>>pip install -r requirements.txt
Run the Streamlit app:


>>>streamlit run app/app.py

What I Learned / Possible Improvements
Solved sklearn version and model serialization challenges

Built and deployed a full ML pipeline

Created a user-friendly interactive interface

Future improvements: add more features, optimize the model, automate retraining
