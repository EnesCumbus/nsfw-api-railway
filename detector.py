import os
import shutil
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_URL = "https://drive.google.com/file/d/13wVvnqAoVssuKDjkIrWPScPhxPvjHuqp"
MODEL_PATH = "saved_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Model indiriliyor...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    del response

model = tf.keras.models.load_model(MODEL_PATH)

# Bu fonksiyonu modeline göre değiştir
CLASS_NAMES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def check_nsfw(image_path):
    image = preprocess_image(image_path)
    preds = model.predict(image)[0]

    result = dict(zip(CLASS_NAMES, preds.tolist()))
    dominant_class = CLASS_NAMES[np.argmax(preds)]
    nsfw_score = result.get("porn", 0.0) + result.get("sexy", 0.0)

    return {
        "safe": nsfw_score <= 0.6,
        "nsfw_score": nsfw_score,
        "dominant_class": dominant_class,
        "scores": result
    }
