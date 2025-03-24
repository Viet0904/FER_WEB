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
import mediapipe as mp
import threading
import time
import queue
import traceback

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Cấu hình logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_detection.log"),
        logging.StreamHandler()
    ]
)

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

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Khởi tạo face_mesh một lần ở mức global
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)  

# Giữ lại Haar cascade như một phương án dự phòng
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Biến toàn cục cho threading
frame_queue = queue.Queue(maxsize=1)  # chỉ giữ frame mới nhất
result_queue = queue.Queue(maxsize=1)  # chỉ giữ kết quả mới nhất
is_processing = False
processing_thread = None

class ViTBasePatch16_224_Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(ViTBasePatch16_224_Model, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

model = None

def load_model(model_path="ViTBase_RAFDB_f1.pth"):
    global model
    if model is None:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model = ViTBasePatch16_224_Model(num_classes=7)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            logging.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    return model

# Load mô hình chính
try:
    model = load_model()
    logging.info("ViT model loaded successfully")
except Exception as e:
    logging.error(f"Error loading ViT model: {str(e)}")
    model = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face_mediapipe(image):
    """Phát hiện khuôn mặt sử dụng MediaPipe Face Mesh"""
    try:
        if isinstance(image, np.ndarray):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = np.array(Image.open(image).convert("RGB"))
            
        # Reset face_mesh cho mỗi frame để tránh lỗi
        results = face_mesh.process(rgb_image)
        
        faces = []
        landmarks = []
        
        if results.multi_face_landmarks:
            h, w = rgb_image.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                # Lấy bounding box từ landmarks
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                face_landmarks_list = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face_landmarks_list.append([x, y])  # Thay đổi từ tuple sang list để đảm bảo có thể truy cập theo chỉ số
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Thêm margin cho bounding box
                margin = int((x_max - x_min) * 0.1)
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
                landmarks.extend(face_landmarks_list)
                
        return faces, landmarks
    except Exception as e:
        logging.error(f"Error in detect_face_mediapipe: {str(e)}\n{traceback.format_exc()}")
        return [], []

def predict_emotion(image, model, return_probs=False, use_mediapipe=True):
    """Nhận diện cảm xúc từ hình ảnh"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
            
        if isinstance(image, np.ndarray):
            original_image = image.copy()
        else:
            original_image = np.array(Image.open(image).convert("RGB"))
            
        # Thử phát hiện khuôn mặt bằng MediaPipe trước
        if use_mediapipe:
            faces, landmarks = detect_face_mediapipe(original_image)
        else:
            faces = []
            landmarks = []
            
        # Sử dụng Haar Cascade nếu MediaPipe không tìm thấy khuôn mặt
        if not faces:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = [(x, y, w, h) for (x, y, w, h) in haar_faces]
            landmarks = []  # Không có landmarks với Haar Cascade
            
        results = []
        if len(faces) == 0:
            return [], landmarks if return_probs else []

        for i, (x, y, w, h) in enumerate(faces):
            # Kiểm tra kích thước khuôn mặt
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > original_image.shape[1] or y + h > original_image.shape[0]:
                logging.warning(f"Invalid face dimensions: x={x}, y={y}, w={w}, h={h}, image shape: {original_image.shape}")
                continue
                
            face = original_image[y:y + h, x:x + w]
            if face.size == 0 or face.shape[0] <= 0 or face.shape[1] <= 0:
                logging.warning(f"Empty face region")
                continue
            
            try:
                face_pil = Image.fromarray(face)
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy().tolist()
                    _, predicted = torch.max(outputs, 1)
                    emotion_idx = predicted.item()
                    
                    # Chỉ sử dụng tên cảm xúc tiếng Anh
                    emotion = label_to_emotion.get(emotion_idx, "Unknown")
                    
                    result = {"emotion": emotion, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                    
                    if return_probs:
                        result["probabilities"] = {label_to_emotion[i]: prob for i, prob in enumerate(probabilities)}
                    
                    results.append(result)
            except Exception as inner_e:
                logging.error(f"Error processing face: {str(inner_e)}\n{traceback.format_exc()}")
                continue
                
        return results, landmarks if return_probs else results

    except Exception as e:
        logging.error(f"Error in predict_emotion: {str(e)}\n{traceback.format_exc()}")
        error_msg = f"Error: {str(e)}"
        return [{"emotion": error_msg, "x": 0, "y": 0, "w": 0, "h": 0}], []

def process_frames():
    """Hàm thread để xử lý frames"""
    global is_processing
    is_processing = True
    
    while is_processing:
        try:
            if not frame_queue.empty():
                # Lấy frame từ queue
                frame = frame_queue.get(block=False)
                
                # Kiểm tra frame
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    logging.warning("Empty or invalid frame received, skipping")
                    time.sleep(0.01)
                    continue
                
                # Đảm bảo frame có đúng định dạng và kích thước
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logging.warning(f"Invalid frame format: {frame.shape}")
                    time.sleep(0.01)
                    continue
                
                # Xử lý ở độ phân giải thấp hơn để tăng tốc
                try:
                    scale_factor = 0.5
                    frame_small = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                except Exception as resize_error:
                    logging.error(f"Error resizing frame: {str(resize_error)}")
                    time.sleep(0.01)
                    continue
                
                # Dự đoán cảm xúc
                try:
                    emotions, landmarks = predict_emotion(
                        frame_small, 
                        model,
                        use_mediapipe=True
                    )
                except Exception as predict_error:
                    logging.error(f"Error in predict_emotion: {str(predict_error)}\n{traceback.format_exc()}")
                    time.sleep(0.01)
                    continue
                
                # Xử lý kết quả
                try:
                    # Scale landmarks trở lại độ phân giải gốc
                    scaled_landmarks = []
                    
                    # FIX: Kiểm tra kiểu dữ liệu của landmarks và xử lý đúng cách
                    for lm in landmarks:
                        try:
                            # Kiểm tra xem lm có phải là dictionary hay không
                            if isinstance(lm, dict) and 0 in lm and 1 in lm:
                                scaled_landmarks.append([int(lm[0] / scale_factor), int(lm[1] / scale_factor)])
                            # Kiểm tra xem lm có phải là list/tuple hay không và có đủ phần tử
                            elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                                scaled_landmarks.append([int(lm[0] / scale_factor), int(lm[1] / scale_factor)])
                            else:
                                logging.warning(f"Invalid landmark format: {type(lm)}")
                        except Exception as lm_error:
                            logging.error(f"Error processing landmark: {str(lm_error)}")
                            continue
                    
                    # Scale bounding box của cảm xúc
                    for emotion in emotions:
                        emotion["x"] = int(emotion["x"] / scale_factor)
                        emotion["y"] = int(emotion["y"] / scale_factor)
                        emotion["w"] = int(emotion["w"] / scale_factor)
                        emotion["h"] = int(emotion["h"] / scale_factor)
                    
                    # Đưa kết quả vào queue
                    if not result_queue.full():
                        while not result_queue.empty():
                            result_queue.get()
                        result_queue.put((emotions, scaled_landmarks))
                except Exception as result_error:
                    logging.error(f"Error processing results: {str(result_error)}\n{traceback.format_exc()}")
            else:
                # Ngủ một chút nếu không có frame để xử lý
                time.sleep(0.01)
                
        except Exception as e:
            logging.error(f"Error in processing thread: {str(e)}\n{traceback.format_exc()}")
            time.sleep(0.1)  # Ngủ khi có lỗi để tránh vòng lặp chặt

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received predict request")
    if "file" not in request.files:
        logging.warning("No file part in request")
        return jsonify({"error": "No file part"})
        
    file = request.files["file"]
    if file.filename == "":
        logging.warning("No selected file")
        return jsonify({"error": "No selected file"})
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        try:
            file.save(filepath)
            logging.info(f"File saved at {filepath}")
            # Sử dụng tiếng Anh cho dự đoán qua upload
            results, landmarks = predict_emotion(filepath, model, return_probs=True, use_mediapipe=True)
            logging.info(f"Prediction result: {results}")
            return jsonify({"emotions": results, "landmarks": landmarks})
        except Exception as e:
            logging.error(f"Error saving or processing file: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Error processing file: {str(e)}"})
            
    logging.warning("Invalid file format")
    return jsonify({"error": "Invalid file format"})

@socketio.on("image")
def handle_image(data):
    try:
        if not isinstance(data, str) or "," not in data:
            socketio.emit("emotion_result", {
                "emotions": [{"emotion": "Error: Invalid image data", "x": 0, "y": 0, "w": 0, "h": 0}],
                "landmarks": []
            })
            return
            
        base64_str = data.split(",")[1]
        image_data = base64.b64decode(base64_str)
        npimg = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            socketio.emit("emotion_result", {
                "emotions": [{"emotion": "Error: Failed to decode image", "x": 0, "y": 0, "w": 0, "h": 0}],
                "landmarks": []
            })
            return
            
        # Thêm frame vào queue để xử lý
        try:
            if not frame_queue.full():
                # Xóa queue trước để tránh xử lý frames cũ
                while not frame_queue.empty():
                    frame_queue.get()
                frame_queue.put(frame)
        except Exception as queue_error:
            logging.error(f"Error putting frame in queue: {str(queue_error)}\n{traceback.format_exc()}")
        
        # Kiểm tra nếu có kết quả mới
        if not result_queue.empty():
            try:
                emotions, landmarks = result_queue.get()
                socketio.emit("emotion_result", {"emotions": emotions, "landmarks": landmarks})
            except Exception as result_error:
                logging.error(f"Error getting results from queue: {str(result_error)}\n{traceback.format_exc()}")
            
    except Exception as e:
        logging.error(f"Error in handle_image: {str(e)}\n{traceback.format_exc()}")
        socketio.emit("emotion_result", {
            "emotions": [{"emotion": f"Error: {str(e)}", "x": 0, "y": 0, "w": 0, "h": 0}],
            "landmarks": []
        })

@socketio.on("connect")
def handle_connect():
    global processing_thread, is_processing
    
    if processing_thread is None or not processing_thread.is_alive():
        is_processing = True
        processing_thread = threading.Thread(target=process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        logging.info("Started processing thread")

@socketio.on("disconnect")
def handle_disconnect():
    global is_processing
    is_processing = False
    logging.info("Stopped processing thread")

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)