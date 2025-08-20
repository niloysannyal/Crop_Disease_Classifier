from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import uvicorn
import os
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="Crops Disease Classification API")


# Crop models
CROPS = {
    "corn": {
        "model_path": "./corn/saved_models/corn_disease_model.h5",
        "class_names": ["Common_Rust", "Gray_Leaf_Spot", "Healthy", "Northern_Leaf_Blight"],
        "image_size": 224
    },
    "potato": {
        "model_path": "./potato/saved_models/potato_disease_model.h5",
        "class_names": ["Early_Blight", "Healthy", "Late_Blight"],
        "image_size": 224
    },
    "rice": {
        "model_path": "./rice/saved_models/rice_disease_model.h5",
        "class_names": ["Brown_Spot", "Healthy", "Leaf_Blast", "Neck_Blast"],
        "image_size": 224
    },
    "wheat": {
        "model_path": "./wheat/saved_models/wheat_disease_model.h5",
        "class_names": ["Brown_Rust", "Healthy", "Yellow_Rust"],
        "image_size": 224
    },
}


# Load all models at startup
models_dict = {}
for crop_name, info in CROPS.items():
    if os.path.exists(info["model_path"]):
        models_dict[crop_name] = tf.keras.models.load_model(info["model_path"], compile=False)
        print(f"Loaded {crop_name} model.")
    else:
        print(f"Model for {crop_name} not found at {info['model_path']}")


# Read file as image (just ensure 3-channel)
def read_and_preprocess_image(data, image_size):
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = np.expand_dims(image, axis=0)  # add batch dim
    return image_array


@app.post("/predict/")
async def predict(crop: str = Form(...), file: UploadFile = File(...)):
    crop_key = crop.lower()
    if crop_key not in models_dict:
        return JSONResponse(content={"error": f"No model found for {crop}"}, status_code=404)
    try:
        # Retrieve crop info
        image_size = CROPS[crop_key]["image_size"]
        class_names = CROPS[crop_key]["class_names"]
        model = models_dict[crop_key]

        # Read and preprocess image
        image_array = read_and_preprocess_image(await file.read(), image_size)

        # Predict
        predictions = model.predict(image_array)
        index = np.argmax(predictions[0])
        predicted_class = class_names[index]
        confidence = float(np.max(predictions[0]))

        return {
            "crop": crop,
            "predicted_class": predicted_class,
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
