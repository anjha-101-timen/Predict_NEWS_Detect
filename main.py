# News Prediction Detection

# Library Imports
import streamlit as st
import numpy as up
import pandas as pd
import joblib
import os
import string


# Model 'Dump' & 'Load' Function with 'Directory Path' & 'FileName'
def dump_model(model, directory_path, filename):
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, filename)
    joblib.dump(model, file_path)

def load_model(directory_path, filename):
    file_path = os.path.join(directory_path, filename)
    model = joblib.load(file_path)
    return model


# To Remove all the Punctuation & Symbols from the 'NEWS Data'
def punctuation_removal(text):
    lit = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(lit)
    return clean_str


# To Remove all the StopWords from the 'NEWS Data'
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopper = stopwords.words("english")


# Name of the Trained Piped Models
news_predict_model = \
    [
        "Logistic Regression",
        "Naive Bayes",
        "Support Vector Machine (SVM)",
        "SGD Classifer",
        "Ridge Classifer",
        "Perceptron",
        "K-Nearest Neighbors (KNN)",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting (GBM)",
        "XG Boost",
        "Light GBM",
        "Ada Boost",
        "Bagging"
    ]


# Function to Predict the 'NEWS' with various Models
def predict_news_with_models(models, data):
    mpredictions = []
    for model in models:
        prediction = model.predict(data)
        mpredictions.append(prediction)
    return mpredictions


# 'NEWS Data' Cleaning Preparation
def cleaned_news(news):
    news_lower = news.apply(lambda z: z.lower())
    news_removed_punctuation = news_lower.apply(punctuation_removal)
    news_cleaned = news_removed_punctuation.apply(lambda z: ' '.join([words for words in z.split() if words not in (stopper)]))
    return news_cleaned


# Function to Predict the 'NEWS' with various Models
def detect_news_predict (models, data) :
    prediction = predict_news_with_models(models, data)
    fake_count = sum(row[0] == 'fake' for row in prediction)
    real_count = sum(row[0] == 'real' for row in prediction)

    confusion_threshold = 2
    if abs(fake_count - real_count) <= confusion_threshold:
        overall_prediction = "Confused"
    elif fake_count > real_count:
        overall_prediction = "Fake NEWS"
    else:
        overall_prediction = "Real NEWS"

    for mpredict in range(0, 14):
        print(f"The NEWS Predicted by the {news_predict_model[mpredict]} Model :  ", prediction[mpredict][0], "!")

    return overall_prediction


# Function to Detect (Prediction) whether the 'NEWS' is 'Fake' or 'Real'
def main() :
    st.title("Real & Fake News Detector")

    # Directory Path
    current_directory = os.getcwd()
    folder_name = "NEWS_Model"
    folder_path = os.path.join(current_directory, folder_name)

    # Dumping all the Trained Piped Models
    # dump_model(logistic_piped_model, folder_path, "logistic_piped_model.joblib")
    # dump_model(naive_bayes_piped_model, folder_path, "naive_bayes_piped_model.joblib")
    # dump_model(svm_piped_model, folder_path, "svm_piped_model.joblib")
    # dump_model(sgd_piped_model, folder_path, "sgd_piped_model.joblib")
    # dump_model(ridge_piped_model, folder_path, "ridge_piped_model.joblib")
    # dump_model(perceptron_piped_model, folder_path, "perceptron_piped_model.joblib")
    # dump_model(knn_piped_model, folder_path, "knn_piped_model.joblib")
    # dump_model(decision_tree_piped_model, folder_path, "decision_tree_piped_model.joblib")
    # dump_model(random_forest_piped_model, folder_path, "random_forest_piped_model.joblib")
    # dump_model(gbm_piped_model, folder_path, "gbm_piped_model.joblib")
    # dump_model(xgb_piped_model, folder_path, "xgb_piped_model.joblib")
    # dump_model(light_gbm_piped_model, folder_path, "light_gbm_piped_model.joblib")
    # dump_model(ada_boost_piped_model, folder_path, "ada_boost_piped_model.joblib")
    # dump_model(bagging_piped_model, folder_path, "bagging_piped_model.joblib")

    # Loading all the Dumped Trained Piped Models
    logistic_detect = load_model(folder_path, "logistic_piped_model.joblib")
    naive_bayes_detect = load_model(folder_path, "naive_bayes_piped_model.joblib")
    svm_detect = load_model(folder_path, "svm_piped_model.joblib")
    sgd_detect = load_model(folder_path, "sgd_piped_model.joblib")
    ridge_detect = load_model(folder_path, "ridge_piped_model.joblib")
    perceptron_detect = load_model(folder_path, "perceptron_piped_model.joblib")
    knn_detect = load_model(folder_path, "knn_piped_model.joblib")
    decision_tree_detect = load_model(folder_path, "decision_tree_piped_model.joblib")
    random_forest_detect = load_model(folder_path, "random_forest_piped_model.joblib")
    gbm_detect = load_model(folder_path, "gbm_piped_model.joblib")
    xgb_detect = load_model(folder_path, "xgb_piped_model.joblib")
    light_gbm_detect = load_model(folder_path, "light_gbm_piped_model.joblib")
    ada_boost_detect = load_model(folder_path, "ada_boost_piped_model.joblib")
    bagging_detect = load_model(folder_path, "bagging_piped_model.joblib")

    # List of the Trained Piped Models
    news_detect_model = [
        logistic_detect,
        naive_bayes_detect,
        svm_detect,
        sgd_detect,
        ridge_detect,
        perceptron_detect,
        knn_detect,
        decision_tree_detect,
        random_forest_detect,
        gbm_detect,
        xgb_detect,
        light_gbm_detect,
        ada_boost_detect,
        bagging_detect
    ]

    # Input the 'NEWS Data' from the User
    news_text = st.text_area("Enter news text :  ", height=500)

    # Detect NEWS Predict Button
    if st.button("Detect", key="button"):

        if news_text :
            news_data = pd.Series([news_text])
            news_cleaned = cleaned_news(news_data)
            overall_prediction = detect_news_predict(news_detect_model, news_cleaned)

            st.subheader("Overall Prediction:")
            st.success(overall_prediction)
        else :
            st.warning("Please enter news text.")


if __name__ == "__main__":
    main()
