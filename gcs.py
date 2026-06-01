import joblib
import os
from google.cloud import storage


def load_model():

    os.makedirs("temp", exist_ok=True)

    model_path = "temp/xgb_obesity_model.pkl"

    if not os.path.exists(model_path):

        client = storage.Client()

        bucket = client.bucket(
            "tech-challenge-4-obesity"
        )

        blob = bucket.blob(
            "models/xgb_obesity_model.pkl"
        )

        blob.download_to_filename(
            model_path
        )

    return joblib.load(model_path)