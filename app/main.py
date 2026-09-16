from fastapi import FastAPI, File, UploadFile
from app.preprocess import preprocess_image
from app.model import predict_image

app = FastAPI(title="Cats vs Dogs Prediction API")

@app.get("/")
def home():
    return {"message": "Cats vs Dogs API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file's raw bytes
    file_bytes = await file.read()

    # Preprocess into the format the model expects
    image_array = preprocess_image(file_bytes)

    # Get the raw prediction (a number between 0 and 1)
    probability = predict_image(image_array)

    # Apply the same threshold logic from your notebook
    result = "Dog" if probability >= 0.5 else "Cat"

    return {
        "raw_prediction": round(probability, 6),
        "prediction": result
    }