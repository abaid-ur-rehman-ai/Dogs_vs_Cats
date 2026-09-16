from tensorflow.keras.models import load_model
import os
import requests

MODEL_PATH = "model/CatsvsDogs.keras"
MODEL_URL = "https://github.com/abaid-ur-rehman-ai/Dogs_vs_Cats/releases/download/v1.0/CatsvsDogs.keras"
model = None

def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully")

def load_cat_dog_model():
    global model
    if model is None:
        download_model_if_missing()
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    return model

def predict_image(image_array):
    model = load_cat_dog_model()
    prediction = model.predict(image_array)
    return float(prediction[0][0])