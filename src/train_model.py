import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def train(input_file_path, output_file_path):

    # Set experiment (LOCAL MODE)
    mlflow.set_experiment("Model Training")

    # Load dataset
    data = pd.read_csv(input_file_path)

    X = data["text"]
    y = data["label"]

    pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("clf", MultinomialNB())
    ])

    with mlflow.start_run():

        pipeline.fit(X, y)

        y_pred = pipeline.predict(X)

        accuracy = accuracy_score(y, y_pred)

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        joblib.dump(pipeline, output_file_path)

        mlflow.log_param("model", "MultinomialNB")
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="intent_model"
        )

        print("\nModel Training Completed")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Saved at: {output_file_path}")


if __name__ == "__main__":
    train(
        "data/clean_data.csv",
        "model/intent_model.pkl"
    )