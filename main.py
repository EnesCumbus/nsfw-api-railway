from fastapi import FastAPI, UploadFile, File
from detector import check_nsfw
import shutil
import os

app = FastAPI()

@app.post("/check")
async def check_image(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = check_nsfw(temp_path)
    except Exception as e:
        os.remove(temp_path)
        return {"error": str(e)}

    os.remove(temp_path)
    return {
        "safe": result["safe"],
        "nsfw_score": result["nsfw_score"],
        "dominant_class": result["dominant_class"],
        "scores": result["scores"]
    }