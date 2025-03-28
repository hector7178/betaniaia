from fastapi import FastAPI

import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import json
from ultralytics.nn.tasks import DetectionModel 
from ultralytics import YOLO
import torchvision
app = FastAPI()
model = YOLO("best.pt")



@app.get("/")
async def root():
    return {"message": "¡Hola, mundo!"}


# Cargar las clases (ajusta según tu configuración)
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Transformaciones para la imagen
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/predict/")
async def predict():
    model = YOLO("best.pt")
    results = model("OIP.jpg")
    for result in results:
        
        confidence = result.boxes # Confianza (0.0 a 1.0)
        class_name = model.names # Nombre de la clase
        
        print(f"Clase: {class_name}, Confianza: {confidence.data.tolist()}")
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg") 
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer y preprocesar la imagen
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert("RGB")  # Asegurar formato RGB

        # Realizar inferencia
        results = model(image)
        
        # Procesar resultados
        detections = []
        for result in results:
            for box in result.boxes:
                # Extraer datos
                xyxy = box.xyxy[0].tolist()  # Coordenadas [x1, y1, x2, y2]
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                
                # Mapear ID de clase a nombre
                class_name = classes[class_id] if class_id < len(classes) else "unknown"
                
                # Agregar detección
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2)
                    }
                })
        
        return {"detections": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))