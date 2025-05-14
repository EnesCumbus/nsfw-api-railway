import os
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "saved_model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=13wVvnqAoVssuKDjkIrWPScPhxPvjHuqp"

# 🔍 İndirilen dosya gerçekten inmiş mi kontrol et
if not os.path.exists(MODEL_PATH):
    print("✅ Model bulunamadı, indiriliyor...")
    gdown.download(url=GDRIVE_URL, output=MODEL_PATH, quiet=False)
else:
    print("✅ Model zaten mevcut, yeniden indirmeye gerek yok.")

# 🧪 Dosya var mı ve boyutu ne kadar?
print("📦 Dosya mevcut mu:", os.path.exists(MODEL_PATH))
if os.path.exists(MODEL_PATH):
    print("📏 Dosya boyutu (byte):", os.path.getsize(MODEL_PATH))

# ✅ Model yükleniyor
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("🚀 Model başarıyla yüklendi.")
except Exception as e:
    print("❌ Model yüklenemedi:", str(e))
    raise

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
