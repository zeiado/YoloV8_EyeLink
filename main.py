from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load your YOLOv8 model
model = YOLO('yolov8n.pt')  # (you can choose other models like yolov8s, yolov8m depending on your need)

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })

    return JSONResponse(content={"detections": detections})
