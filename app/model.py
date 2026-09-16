from tensorflow.keras.models import load_model

MODEL_PATH = "model/CatsvsDogs.keras"
model = None

def load_cat_dog_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    return model

def predict_image(image_array):
    model = load_cat_dog_model()
    prediction = model.predict(image_array)
    return float(prediction[0][0])