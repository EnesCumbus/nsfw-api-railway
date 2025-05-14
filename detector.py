import os
import shutil
import requests
from nsfw_detector import predict

MODEL_URL = "https://drive.google.com/file/d/13wVvnqAoVssuKDjkIrWPScPhxPvjHuqp"
MODEL_PATH = "saved_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    del response

model = predict.load_model(MODEL_PATH)

def check_nsfw(image_path):
    result = predict.classify(model, image_path)
    scores = result[image_path]
    nsfw_score = scores.get("porn", 0.0) + scores.get("sexy", 0.0)
    dominant_class = max(scores, key=scores.get)
    return {
        "nsfw_score": nsfw_score,
        "dominant_class": dominant_class,
        "scores": scores,
        "safe": nsfw_score <= 0.6
    }
