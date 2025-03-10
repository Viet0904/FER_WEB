# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import timm
import cv2
import base64
from io import BytesIO
import logging
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_to_emotion = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Neutral"}
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = None
def load_model(model_path=r"D:\Workspace\CTU\Web_LuanVan_Demo\ViTBase_RAFDB_f1.pth"):
    global model
    if model is None:
        logger.debug("Loading model...")
        model = ViTBasePatch16_224_Model(num_classes=7)
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        logger.debug("Model loaded successfully")
    return model

class ViTBasePatch16_224_Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(ViTBasePatch16_224_Model, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

model = load_model()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_emotion(image, model, return_probs=False):
    try:
        start_time = datetime.utcnow()
        if isinstance(image, np.ndarray):
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            original_image = np.array(Image.open(image).convert("RGB"))

        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))  # Tăng độ nhạy
        logger.debug(f"Detected faces at {datetime.utcnow().isoformat()}Z: {len(faces)} with coordinates: {faces}")

        results = []
        if len(faces) == 0:
            logger.debug("No faces detected, returning empty results")
            return results if not return_probs else {"emotion": "No face detected", "probabilities": {}}

        for x, y, w, h in faces:
            face = original_image[y:y + h, x:x + w]
            if face.size == 0:
                logger.warning(f"Empty face region at ({x}, {y}, {w}, {h})")
                continue

            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                emotion_idx = predicted.item()
                emotion = label_to_emotion.get(emotion_idx, "Unknown")

            result = {"emotion": emotion, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            logger.debug(f"Added result: {result}")
            results.append(result)

        logger.debug(f"Final results at {datetime.utcnow().isoformat()}Z: {results}")
        logger.debug(f"Processing time: {(datetime.utcnow() - start_time).total_seconds()} seconds")
        return results

    except Exception as e:
        logger.error(f"Error in predict_emotion: {str(e)}", exc_info=True)
        return [{"emotion": f"Error: {str(e)}", "x": 0, "y": 0, "w": 0, "h": 0}]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        result = predict_emotion(filepath, model, return_probs=True)
        return jsonify(result)
    return jsonify({"error": "Invalid file format"})

@socketio.on("image")
def handle_image(data):
    try:
        base64_str = data.split(",")[1]
        image_data = base64.b64decode(base64_str)
        npimg = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Failed to decode image")
            socketio.emit("emotion_result", {"emotions": [{"emotion": "Error: Failed to decode image", "x": 0, "y": 0, "w": 0, "h": 0}]})
            return

        emotions = predict_emotion(frame, model)
        logger.debug(f"Sending emotions at: {datetime.utcnow().isoformat()}Z, {emotions}")
        socketio.emit("emotion_result", {"emotions": emotions})
    except Exception as e:
        logger.error(f"Error in handle_image: {str(e)}", exc_info=True)
        socketio.emit("emotion_result", {"emotions": [{"emotion": f"Error: {str(e)}", "x": 0, "y": 0, "w": 0, "h": 0}]})

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, debug=True)