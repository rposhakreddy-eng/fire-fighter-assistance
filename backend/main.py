from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from detector import detect_image

import shutil
import os


app = FastAPI(
    title="Firefighter Assistance Thermal Detection"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():

    return {
        "message": "Thermal Victim Detection AI is running"
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    os.makedirs("../results", exist_ok=True)

    image_path = "../results/uploaded_image.jpg"

    with open(image_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    detections = detect_image(image_path)


    if len(detections) == 0:

        return {
            "result": "No human detected",
            "confidence": 0,
            "detected_object": "None"
        }


    best_detection = max(
        detections,
        key=lambda x: x["confidence"]
    )


    return {
        "result": "Possible human detected",
        "confidence": round(
            best_detection["confidence"] * 100,
            2
        ),
        "detected_object":
            best_detection["object"]
    }