import json
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse

# Global dictionary to hold our model and mappings in memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model and class mappings from the models/ directory...")
    
    # 1. Load ONNX Session (It will automatically find the .data file in the same folder)
    ml_models["session"] = ort.InferenceSession(
        "models/nano_ocr.onnx", 
        providers=["CPUExecutionProvider"] # Optimized for your i5 CPU
    )
    
    # 2. Load the JSON dictionary
    with open("models/class_mapping.json", "r") as f:
        ml_models["class_mapping"] = json.load(f)
        
    yield # API is running
    ml_models.clear() # Clean up memory on shutdown

app = FastAPI(title="NanoOCR API", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r") as f:
        return f.read()

def preprocess_crop(crop_img: np.ndarray) -> np.ndarray:
    """Prepares a single OpenCV crop to match our Colab PyTorch transforms."""
    # 1. Resize to 32x32
    resized = cv2.resize(crop_img, (32, 32))
    
    # 2. Convert to float32 and scale to [0, 1]
    img_arr = resized.astype(np.float32) / 255.0
    
    # 3. Normalize to [-1, 1] exactly like transforms.Normalize((0.5,), (0.5,))
    img_arr = (img_arr - 0.5) / 0.5
    
    # 4. Reshape to (Channel=1, Height=32, Width=32)
    return np.expand_dims(img_arr, axis=0)

@app.post("/read_document")
async def read_document(file: UploadFile = File(...)):
    # Ensure the user uploaded an image
    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="File must be an image.")
         
    # 1. Read Image into OpenCV directly from memory (no saving to disk)
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = image.shape
    
    # 2. Text Detection (Binarization & Contours)
    # Blur to remove noise, then apply inverse threshold (black background, white text)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours (the borders of the letters)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter and create bounding boxes
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out tiny specks of noise (dust on the camera/page)
        if w > 5 and h > 10: 
            bounding_boxes.append((x, y, w, h))
            
    # Sort bounding boxes strictly from Left to Right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    # 4. Batch Preprocessing
    batch_tensors = []
    for (x, y, w, h) in bounding_boxes:
        # Crop the letter and add a small 2px padding, respecting image boundaries
        crop = image[max(0, y-2):min(img_h, y+h+2), max(0, x-2):min(img_w, x+w+2)]
        tensor = preprocess_crop(crop)
        batch_tensors.append(tensor)
        
    if not batch_tensors:
        return {"result": "No text detected."}
        
    # Stack crops into a single batch tensor for ONNX: Shape -> (Batch, 1, 32, 32)
    batch_input = np.stack(batch_tensors, axis=0)
    
    # 5. Text Recognition (Blazing fast CPU Batch Inference)
    session = ml_models["session"]
    input_name = session.get_inputs()[0].name
    
    # Run the model!
    outputs = session.run(None, {input_name: batch_input})[0] # Returns shape (Batch, 36)
    
    # 6. Decode Predictions back into a string
    predicted_indices = np.argmax(outputs, axis=1)
    reconstructed_text = "".join(
        [ml_models["class_mapping"].get(str(idx), "") for idx in predicted_indices]
    )
    
    return {
        "filename": file.filename,
        "characters_detected": len(bounding_boxes),
        "extracted_text": reconstructed_text
    }