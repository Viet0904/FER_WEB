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
import sys

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Cấu hình logging với encoding UTF-8
import logging


class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("emotion_detection.log", encoding="utf-8"),
        UTF8StreamHandler(sys.stdout),
    ],
)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_to_emotion = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral",
}

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=100,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
is_processing = False
processing_thread = None


class ViTBasePatch16_224_Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(ViTBasePatch16_224_Model, self).__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


model = None


def load_model(model_path="ViTBase_RAFDB_f1.pth"):
    global model
    if model is None:
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
            model = ViTBasePatch16_224_Model(num_classes=7)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
            model.load_state_dict(new_state_dict)
            model.to(device)
            model.eval()
            logging.info(f"Đã tải mô hình thành công từ {model_path}")
        except Exception as e:
            logging.error(f"Lỗi khi tải mô hình: {str(e)}")
            raise
    return model


try:
    model = load_model()
    logging.info("Đã tải mô hình ViT thành công")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình ViT: {str(e)}")
    model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_face_mediapipe(image, max_faces=1):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        faces = []
        landmarks = []
        if results.multi_face_landmarks:
            h, w = rgb_image.shape[:2]
            for face_landmarks in results.multi_face_landmarks[:max_faces]:
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                face_landmarks_list = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face_landmarks_list.append([x, y])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                margin = int((x_max - x_min) * 0.1)
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
                landmarks.extend(face_landmarks_list)
        return faces, landmarks
    except Exception as e:
        logging.error(
            f"Lỗi trong detect_face_mediapipe: {str(e)}\n{traceback.format_exc()}"
        )
        return [], []


def predict_emotion(image, model, max_faces=1, return_probs=False):
    try:
        if model is None:
            raise ValueError("Mô hình chưa được tải")
        original_image = image.copy()
        faces, landmarks = detect_face_mediapipe(original_image, max_faces=max_faces)
        if not faces:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            haar_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            faces = [(x, y, w, h) for (x, y, w, h) in haar_faces][:max_faces]
            landmarks = []
        results = []
        if len(faces) == 0:
            return results, landmarks  # Luôn trả về cả results và landmarks
        for i, (x, y, w, h) in enumerate(faces):
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > original_image.shape[1]
                or y + h > original_image.shape[0]
            ):
                logging.warning(
                    f"Kích thước khuôn mặt không hợp lệ: x={x}, y={y}, w={w}, h={h}"
                )
                continue
            face = original_image[y : y + h, x : x + w]
            if face.size == 0 or face.shape[0] <= 0 or face.shape[1] <= 0:
                logging.warning(f"Vùng khuôn mặt trống")
                continue
            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy().tolist()
                _, predicted = torch.max(outputs, 1)
                emotion_idx = predicted.item()
                emotion = label_to_emotion.get(emotion_idx, "Unknown")
                result = {
                    "emotion": emotion,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                }
                if return_probs:
                    result["probabilities"] = {
                        label_to_emotion[i]: prob
                        for i, prob in enumerate(probabilities)
                    }
                results.append(result)
        return results, landmarks  # Luôn trả về cả results và landmarks
    except Exception as e:
        logging.error(f"Lỗi trong predict_emotion: {str(e)}\n{traceback.format_exc()}")
        return [{"emotion": f"Lỗi: {str(e)}", "x": 0, "y": 0, "w": 0, "h": 0}], []


def process_frames():
    global is_processing
    is_processing = True
    while is_processing:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get(block=False)
                if (
                    frame is None
                    or not isinstance(frame, np.ndarray)
                    or frame.size == 0
                ):
                    logging.warning("Nhận được frame trống hoặc không hợp lệ, bỏ qua")
                    time.sleep(0.01)
                    continue
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logging.warning(f"Định dạng frame không hợp lệ: {frame.shape}")
                    time.sleep(0.01)
                    continue
                scale_factor = 0.5
                frame_small = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                emotions, _ = predict_emotion(
                    frame_small, model, max_faces=1
                )  # Bỏ landmarks
                for emotion in emotions:
                    emotion["x"] = int(emotion["x"] / scale_factor)
                    emotion["y"] = int(emotion["y"] / scale_factor)
                    emotion["w"] = int(emotion["w"] / scale_factor)
                    emotion["h"] = int(emotion["h"] / scale_factor)
                if not result_queue.full():
                    while not result_queue.empty():
                        result_queue.get()
                    result_queue.put({"emotions": emotions})  # Chỉ gửi emotions
            else:
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Lỗi trong thread xử lý: {str(e)}\n{traceback.format_exc()}")
            time.sleep(0.1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Nhận được yêu cầu dự đoán")
    if "file" not in request.files:
        logging.warning("Không có phần file trong yêu cầu")
        return jsonify({"error": "Không có phần file"})
    file = request.files["file"]
    if file.filename == "":
        logging.warning("Không có file được chọn")
        return jsonify({"error": "Không có file được chọn"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            file.save(filepath)
            logging.info(f"File đã được lưu tại {filepath}")
            img = cv2.imread(filepath)
            results, landmarks = predict_emotion(
                img, model, max_faces=10, return_probs=True
            )
            logging.info(f"Kết quả dự đoán: {results}")
            return jsonify({"emotions": results, "landmarks": landmarks})
        except Exception as e:
            logging.error(
                f"Lỗi khi lưu hoặc xử lý file: {str(e)}\n{traceback.format_exc()}"
            )
            return jsonify({"error": f"Lỗi khi xử lý file: {str(e)}"})
    logging.warning("Định dạng file không hợp lệ")
    return jsonify({"error": "Định dạng file không hợp lệ"})


@socketio.on("image")
def handle_image(data):
    try:
        if not isinstance(data, str) or "," not in data:
            socketio.emit(
                "emotion_result",
                {
                    "emotions": [
                        {
                            "emotion": "Lỗi: Dữ liệu hình ảnh không hợp lệ",
                            "x": 0,
                            "y": 0,
                            "w": 0,
                            "h": 0,
                        }
                    ],
                    "landmarks": [],
                },
            )
            return
        base64_str = data.split(",")[1]
        image_data = base64.b64decode(base64_str)
        npimg = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            socketio.emit(
                "emotion_result",
                {
                    "emotions": [
                        {
                            "emotion": "Lỗi: Không thể giải mã hình ảnh",
                            "x": 0,
                            "y": 0,
                            "w": 0,
                            "h": 0,
                        }
                    ],
                    "landmarks": [],
                },
            )
            return
        if not frame_queue.full():
            while not frame_queue.empty():
                frame_queue.get()
            frame_queue.put(frame)
        if not result_queue.empty():
            result = result_queue.get()
            socketio.emit("emotion_result", result)
    except Exception as e:
        logging.error(f"Lỗi trong handle_image: {str(e)}\n{traceback.format_exc()}")
        socketio.emit(
            "emotion_result",
            {
                "emotions": [
                    {"emotion": f"Lỗi: {str(e)}", "x": 0, "y": 0, "w": 0, "h": 0}
                ],
                "landmarks": [],
            },
        )


@socketio.on("connect")
def handle_connect():
    global processing_thread, is_processing
    if processing_thread is None or not processing_thread.is_alive():
        is_processing = True
        processing_thread = threading.Thread(target=process_frames)
        processing_thread.daemon = True
        processing_thread.start()
        logging.info("Đã khởi động thread xử lý")


@socketio.on("disconnect")
def handle_disconnect():
    global is_processing
    is_processing = False
    logging.info("Đã dừng thread xử lý")


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
