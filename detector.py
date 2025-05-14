import os
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "saved_model.h5"
MODEL_URL = "https://huggingface.co/eneswayn/nsfw-model/resolve/main/saved_model.h5"

# Model dosyası yoksa indir
if not os.path.exists(MODEL_PATH):
    print("✅ Model bulunamadı, indiriliyor...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model indirildi.")

# Modeli yükle
model = tf.keras.models.load_model(MODEL_PATH)

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
