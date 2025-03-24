<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Emotion Detection</title>
    <style>
      body {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f7f7f7;
      }
      h1 {
        color: #333;
        text-align: center;
        margin-bottom: 30px;
      }
      .container {
        display: flex;
        flex-direction: column;
        gap: 20px;
      }
      .tab {
        display: none;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      .tab.active {
        display: block;
      }
      .tabs {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        justify-content: center;
      }
      .tab-btn {
        padding: 12px 24px;
        background-color: #f0f0f0;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s;
      }
      .tab-btn:hover {
        background-color: #e0e0e0;
      }
      .tab-btn.active {
        background-color: #4285f4;
        color: white;
      }
      .video-container {
        position: relative;
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
      }
      #video,
      #canvas {
        width: 100%;
        max-width: 640px;
        height: auto;
        border-radius: 8px;
      }
      #canvas {
        position: absolute;
        top: 0;
        left: 0;
      }
      #landmarksCanvas {
        position: absolute;
        top: 0;
        left: 0;
        z-index: 10;
      }
      .emotion-label {
        position: absolute;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px;
        border-radius: 3px;
        font-size: 14px;
      }
      .btn {
        padding: 12px 24px;
        background-color: #4285f4;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: 600;
        transition: all 0.2s;
        display: inline-block;
        margin: 10px 0;
      }
      .btn:hover {
        background-color: #3367d6;
        transform: translateY(-2px);
      }
      .btn:active {
        transform: translateY(0);
      }
      .btn-group {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin: 15px 0;
      }
      #uploadPreview {
        max-width: 100%;
        max-height: 600px;
        width: auto;
        height: auto;
        display: none;
        margin: 15px auto;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        object-fit: contain;
      }
      .preview-container {
        width: 100%;
        text-align: center;
        margin: 15px 0;
        min-height: 300px;
      }
      .result-container {
        margin-top: 20px;
        padding: 15px;
        background-color: #f9f9f9;
        border-radius: 8px;
        border-left: 4px solid #4285f4;
      }
      .progress-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin-top: 8px;
        overflow: hidden;
      }
      .progress-bar {
        height: 20px;
        border-radius: 5px;
        background-color: #4285f4;
        text-align: center;
        color: white;
        font-size: 12px;
        line-height: 20px;
        transition: width 0.3s ease;
      }
      .file-input-wrapper {
        position: relative;
        overflow: hidden;
        display: inline-block;
        margin: 10px 0;
      }
      .file-input-wrapper input[type="file"] {
        font-size: 100px;
        position: absolute;
        left: 0;
        top: 0;
        opacity: 0;
        cursor: pointer;
      }
      .file-input-wrapper .btn {
        display: inline-block;
        padding: 12px 24px;
      }
      .camera-options {
        margin: 15px 0;
        text-align: center;
        white-space: nowrap;
      }
      .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        display: none;
      }
      .loading {
        display: none;
        text-align: center;
        margin: 15px 0;
      }
      .loading::after {
        content: "⏳";
        animation: loading 1.5s infinite;
        font-size: 20px;
        margin-left: 10px;
      }
      @keyframes loading {
        0% {
          opacity: 0.3;
        }
        50% {
          opacity: 1;
        }
        100% {
          opacity: 0.3;
        }
      }
      #cameraSelection {
        padding: 8px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-left: 10px;
      }
      .upload-buttons-container {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        gap: 10px;
        margin-bottom: 15px;
      }
    </style>
  </head>
  <body>
    <h1>Emotion Detection</h1>

    <div class="tabs">
      <button class="tab-btn active" onclick="showTab('upload')">
        Upload Image
      </button>
      <button class="tab-btn" onclick="showTab('realtime')">
        Real-time Detection
      </button>
    </div>

    <div class="container">
      <div id="uploadTab" class="tab active">
        <h2>Upload Image</h2>
        <div class="upload-buttons-container">
          <div class="file-input-wrapper">
            <button class="btn">Choose Image</button>
            <input
              type="file"
              id="fileInput"
              accept=".jpg,.jpeg,.png"
              onchange="previewImage()"
            />
          </div>
          <button type="button" class="btn" onclick="uploadAndPredict()">
            Upload and Predict
          </button>
        </div>

        <div id="fileInputError" class="error-message"></div>
        <div id="uploadLoading" class="loading">Processing image...</div>

        <div class="preview-container">
          <img id="uploadPreview" alt="Upload Preview" />
        </div>

        <div id="uploadResult" class="result-container"></div>
      </div>

      <div id="realtimeTab" class="tab">
        <h2>Real-time Detection</h2>

        <div class="camera-options">
          <button id="startBtn" class="btn" onclick="toggleWebcam()">
            Start
          </button>
          <button
            id="switchCameraBtn"
            class="btn"
            onclick="switchCamera()"
            style="display: none"
          >
            Switch Camera
          </button>
          <select
            id="cameraSelection"
            style="display: none"
            onchange="changeCamera()"
          >
            <option value="">Select camera</option>
          </select>
        </div>

        <div id="cameraError" class="error-message"></div>

        <div class="video-container">
          <video id="video" autoplay playsinline></video>
          <canvas id="canvas"></canvas>
          <canvas id="landmarksCanvas"></canvas>
        </div>

        <div id="realtimeResult" class="result-container"></div>
      </div>
    </div>

    <script src="https://cdn.socket.io/4.4.1/socket.io.min.js"></script>
    <script>
      let socket;
      let videoStream;
      let isStreaming = false;
      let lastCaptureTime = 0;
      const CAPTURE_INTERVAL_MS = 200;
      let currentCameraIndex = 0;
      let availableCameras = [];

      function initSocket() {
        if (!socket) {
          const protocol =
            window.location.protocol === "https:" ? "wss:" : "ws:";
          const host = window.location.host;
          const socketUrl = `${protocol}//${host}`;
          socket = io(socketUrl);

          socket.on("connect", () => {
            console.log("Connected to server");
          });

          socket.on("emotion_result", (data) => {
            displayRealtimeResult(data);
          });

          socket.on("disconnect", () => {
            console.log("Disconnected from server");
            showError(
              "cameraError",
              "Lost connection to server. Please refresh the page."
            );
          });

          socket.on("connect_error", (error) => {
            console.error("Socket.IO connection error:", error);
            showError(
              "cameraError",
              "Cannot connect to server. Check your network."
            );
          });

          socket.on("error", (error) => {
            console.error("Socket.IO error:", error);
            showError("cameraError", "Socket error: " + error);
          });
        }
      }

      function showError(elementId, message) {
        const errorElement = document.getElementById(elementId);
        errorElement.textContent = message;
        errorElement.style.display = "block";
        setTimeout(() => {
          errorElement.style.display = "none";
        }, 5000);
      }

      function clearError(elementId) {
        const errorElement = document.getElementById(elementId);
        errorElement.style.display = "none";
      }

      function showTab(tabName) {
        document
          .querySelectorAll(".tab")
          .forEach((tab) => tab.classList.remove("active"));
        document
          .querySelectorAll(".tab-btn")
          .forEach((btn) => btn.classList.remove("active"));
        document.getElementById(tabName + "Tab").classList.add("active");
        document
          .querySelector(`.tab-btn[onclick="showTab('${tabName}')"]`)
          .classList.add("active");

        if (tabName === "realtime") {
          initSocket();
          if (!isStreaming && availableCameras.length === 0) {
            checkCameraAvailability();
          }
        } else if (isStreaming) {
          stopWebcam();
        }
      }

      async function checkCameraAvailability() {
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter(
            (device) => device.kind === "videoinput"
          );
          if (videoDevices.length === 0) {
            showError("cameraError", "No camera found on your device.");
            return;
          }
          availableCameras = videoDevices;
          if (videoDevices.length > 1) {
            document.getElementById("switchCameraBtn").style.display =
              "inline-block";
            const cameraSelection = document.getElementById("cameraSelection");
            cameraSelection.innerHTML = "";
            cameraSelection.style.display = "inline-block";
            videoDevices.forEach((device, index) => {
              const option = document.createElement("option");
              option.value = index;
              option.text = device.label || `Camera ${index + 1}`;
              cameraSelection.appendChild(option);
            });
          }
        } catch (error) {
          console.error("Error checking cameras:", error);
          showError(
            "cameraError",
            `Cannot access camera list: ${error.message}`
          );
        }
      }

      function previewImage() {
        const fileInput = document.getElementById("fileInput");
        const preview = document.getElementById("uploadPreview");
        const errorElement = document.getElementById("fileInputError");
        clearError("fileInputError");
        if (fileInput.files && fileInput.files[0]) {
          const file = fileInput.files[0];
          if (
            !file.type.match("image/jpeg") &&
            !file.type.match("image/jpg") &&
            !file.type.match("image/png")
          ) {
            showError("fileInputError", "Please select a JPEG or PNG image.");
            fileInput.value = "";
            preview.style.display = "none";
            return;
          }
          if (file.size > 5 * 1024 * 1024) {
            showError(
              "fileInputError",
              "File size too large. Please select a file smaller than 5MB."
            );
            fileInput.value = "";
            preview.style.display = "none";
            return;
          }
          const reader = new FileReader();
          reader.onload = function (e) {
            const img = new Image();
            img.onload = function () {
              preview.src = e.target.result;
              preview.style.display = "block";
              document.getElementById("uploadResult").innerHTML = "";
            };
            img.src = e.target.result;
          };
          reader.onerror = function () {
            showError(
              "fileInputError",
              "Error reading file. Please try again."
            );
            preview.style.display = "none";
          };
          reader.readAsDataURL(file);
        }
      }

      function uploadAndPredict() {
        const fileInput = document.getElementById("fileInput");
        const uploadLoading = document.getElementById("uploadLoading");
        clearError("fileInputError");
        if (!fileInput.files || !fileInput.files[0]) {
          showError("fileInputError", "Please select an image first.");
          return;
        }
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        document.getElementById("uploadResult").innerHTML = "";
        uploadLoading.style.display = "block";
        fetch("/predict", {
          method: "POST",
          body: formData,
        })
          .then((response) => {
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            return response.json();
          })
          .then((data) => {
            uploadLoading.style.display = "none";
            displayUploadResult(data);
          })
          .catch((error) => {
            console.error("Error:", error);
            uploadLoading.style.display = "none";
            showError("fileInputError", `Processing error: ${error.message}`);
          });
      }

      function displayUploadResult(data) {
        const resultDiv = document.getElementById("uploadResult");
        resultDiv.innerHTML = "";
        if (data.error) {
          let errorMsg;
          switch (data.error) {
            case "No file part":
              errorMsg = "No file part in the request.";
              break;
            case "No selected file":
              errorMsg = "No file selected.";
              break;
            case "Invalid file format":
              errorMsg = "Invalid file format.";
              break;
            default:
              errorMsg = data.error;
          }
          showError("fileInputError", errorMsg);
          return;
        }
        if (!data.emotions || data.emotions.length === 0) {
          resultDiv.innerHTML = "<p>No faces detected in the image.</p>";
          return;
        }
        const resultContainer = document.createElement("div");
        data.emotions.forEach((result, index) => {
          const faceResult = document.createElement("div");
          faceResult.innerHTML = `<h3>Face ${index + 1}: ${
            result.emotion
          }</h3>`;
          if (result.probabilities) {
            const probContainer = document.createElement("div");
            Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .forEach(([emotion, probability]) => {
                const percentage = Math.round(probability * 100);
                probContainer.innerHTML += `
                  <div>
                    <span>${emotion}: ${percentage}%</span>
                    <div class="progress-container">
                      <div class="progress-bar" style="width: ${percentage}%">${percentage}%</div>
                    </div>
                  </div>
                `;
              });
            faceResult.appendChild(probContainer);
          }
          resultContainer.appendChild(faceResult);
        });
        const previewImg = document.getElementById("uploadPreview");
        const canvas = document.createElement("canvas");
        const img = new Image();
        img.onload = function () {
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          if (data.landmarks && data.landmarks.length > 0) {
            ctx.fillStyle = "#00ff00";
            data.landmarks.forEach((point) => {
              ctx.beginPath();
              ctx.arc(point[0], point[1], 1, 0, 2 * Math.PI);
              ctx.fill();
            });
            ctx.strokeStyle = "rgba(0, 255, 0, 0.3)";
            ctx.lineWidth = 0.5;
            connectLandmarks(ctx, data.landmarks);
          }
          data.emotions.forEach((result) => {
            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 2;
            ctx.strokeRect(result.x, result.y, result.w, result.h);
            ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
            ctx.fillRect(result.x, result.y - 25, 100, 25);
            ctx.fillStyle = "#ffffff";
            ctx.font = "16px Arial";
            ctx.fillText(result.emotion, result.x + 5, result.y - 5);
          });
          previewImg.src = canvas.toDataURL();
          resultDiv.appendChild(resultContainer);
        };
        img.src = previewImg.src;
      }

      function connectLandmarks(ctx, landmarks) {
        if (landmarks.length < 30) return;
        const faceRegions = [
          generatePointIndices(0, landmarks.length / 8),
          generatePointIndices(landmarks.length / 8, landmarks.length / 6),
          generatePointIndices(landmarks.length / 6, landmarks.length / 4),
          generatePointIndices(landmarks.length / 4, landmarks.length / 3),
          generatePointIndices(landmarks.length / 3, landmarks.length / 2),
        ];
        const colors = [
          "rgba(255, 0, 0, 0.5)",
          "rgba(0, 255, 0, 0.5)",
          "rgba(0, 0, 255, 0.5)",
          "rgba(255, 255, 0, 0.5)",
          "rgba(255, 0, 255, 0.5)",
        ];
        faceRegions.forEach((region, regionIndex) => {
          ctx.strokeStyle = colors[regionIndex];
          ctx.beginPath();
          for (let i = 0; i < region.length; i++) {
            const idx = region[i];
            if (idx < landmarks.length) {
              if (i === 0) ctx.moveTo(landmarks[idx][0], landmarks[idx][1]);
              else ctx.lineTo(landmarks[idx][0], landmarks[idx][1]);
            }
          }
          if (regionIndex !== 2 && regionIndex !== 3) ctx.closePath();
          ctx.stroke();
        });
      }

      function generatePointIndices(start, end) {
        const indices = [];
        start = Math.floor(start);
        end = Math.floor(end);
        for (let i = start; i < end; i++) indices.push(i);
        return indices;
      }

      function toggleWebcam() {
        if (isStreaming) {
          stopWebcam();
          document.getElementById("startBtn").textContent = "Start";
        } else {
          startWebcam();
        }
      }

      async function startWebcam() {
        clearError("cameraError");
        try {
          if (videoStream)
            videoStream.getTracks().forEach((track) => track.stop());
          if (availableCameras.length === 0) {
            await checkCameraAvailability();
            if (availableCameras.length === 0)
              throw new Error("No camera found.");
          }
          const selectedCamera = availableCameras[currentCameraIndex];
          const constraints = {
            video: {
              deviceId: selectedCamera.deviceId
                ? { exact: selectedCamera.deviceId }
                : undefined,
              width: { ideal: 640 },
              height: { ideal: 480 },
            },
          };
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          const video = document.getElementById("video");
          const canvas = document.getElementById("canvas");
          const landmarksCanvas = document.getElementById("landmarksCanvas");
          videoStream = stream;
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            landmarksCanvas.width = video.videoWidth;
            landmarksCanvas.height = video.videoHeight;
            isStreaming = true;
            document.getElementById("startBtn").textContent = "Stop";
            requestAnimationFrame(captureFrame);
          };
          video.onerror = (e) => {
            console.error("Video error:", e);
            showError(
              "cameraError",
              "Error playing video. Please refresh and try again."
            );
            stopWebcam();
          };
        } catch (error) {
          console.error("Error accessing webcam:", error);
          let errorMessage;
          switch (error.name) {
            case "NotFoundError":
              errorMessage = "No camera found. Check camera connection.";
              break;
            case "NotAllowedError":
              errorMessage =
                "Camera access denied. Please allow camera access.";
              break;
            case "NotReadableError":
            case "AbortError":
              errorMessage =
                "Cannot start camera. It may be in use by another app.";
              break;
            case "OverconstrainedError":
              errorMessage =
                "Camera does not meet requirements. Check settings.";
              break;
            case "SecurityError":
              errorMessage = "Security error accessing camera.";
              break;
            case "TypeError":
              errorMessage = "Invalid data type. Check settings.";
              break;
            default:
              errorMessage = "Unknown error accessing camera: " + error.message;
          }
          showError("cameraError", errorMessage);
          isStreaming = false;
          document.getElementById("startBtn").textContent = "Start";
        }
      }

      function stopWebcam() {
        if (videoStream) {
          videoStream.getTracks().forEach((track) => track.stop());
          document.getElementById("video").srcObject = null;
          isStreaming = false;
          const canvas = document.getElementById("canvas");
          const landmarksCanvas = document.getElementById("landmarksCanvas");
          const canvasCtx = canvas.getContext("2d");
          const landmarksCtx = landmarksCanvas.getContext("2d");
          canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
          landmarksCtx.clearRect(
            0,
            0,
            landmarksCanvas.width,
            landmarksCanvas.height
          );
          document.getElementById("realtimeResult").innerHTML = "";
        }
      }

      function switchCamera() {
        if (availableCameras.length <= 1) {
          showError("cameraError", "No other camera available to switch.");
          return;
        }
        currentCameraIndex = (currentCameraIndex + 1) % availableCameras.length;
        document.getElementById("cameraSelection").value = currentCameraIndex;
        if (isStreaming) {
          stopWebcam();
          startWebcam();
        }
      }

      function changeCamera() {
        const selection = document.getElementById("cameraSelection");
        const newIndex = parseInt(selection.value);
        if (
          !isNaN(newIndex) &&
          newIndex >= 0 &&
          newIndex < availableCameras.length
        ) {
          currentCameraIndex = newIndex;
          if (isStreaming) {
            stopWebcam();
            startWebcam();
          }
        }
      }

      function captureFrame() {
        if (!isStreaming) return;
        const now = Date.now();
        if (now - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
          const video = document.getElementById("video");
          if (video.videoWidth === 0 || video.videoHeight === 0) {
            console.log("Video dimensions not available yet");
            requestAnimationFrame(captureFrame);
            return;
          }
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const imageData = canvas.toDataURL("image/jpeg", 0.7);
          if (socket && socket.connected) {
            socket.emit("image", imageData); // Gửi dữ liệu ảnh mà không cần numFaces
            lastCaptureTime = now;
          }
        }
        requestAnimationFrame(captureFrame);
      }

      function displayRealtimeResult(data) {
        const canvas = document.getElementById("canvas");
        const landmarksCanvas = document.getElementById("landmarksCanvas");
        const resultDiv = document.getElementById("realtimeResult");
        if (!canvas || !landmarksCanvas) return;
        const ctx = canvas.getContext("2d");
        const landmarksCtx = landmarksCanvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        landmarksCtx.clearRect(
          0,
          0,
          landmarksCanvas.width,
          landmarksCanvas.height
        );
        if (!data.emotions || data.emotions.length === 0) {
          resultDiv.innerHTML = "<p>No faces detected</p>";
          return;
        }
        resultDiv.innerHTML = "";
        data.emotions.forEach((result, index) => {
          ctx.strokeStyle = "#00ff00";
          ctx.lineWidth = 3;
          ctx.strokeRect(result.x, result.y, result.w, result.h);
          ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
          ctx.fillRect(result.x, result.y - 25, 130, 25);
          ctx.fillStyle = "#ffffff";
          ctx.font = "16px Arial";
          ctx.fillText(result.emotion, result.x + 5, result.y - 5);
          const faceResult = document.createElement("div");
          faceResult.innerHTML = `<h3>Face ${index + 1}: ${
            result.emotion
          }</h3>`;
          resultDiv.appendChild(faceResult);
        });
        if (data.landmarks && data.landmarks.length > 0) {
          landmarksCtx.fillStyle = "#00ff00";
          data.landmarks.forEach((point) => {
            landmarksCtx.beginPath();
            landmarksCtx.arc(point[0], point[1], 1, 0, 2 * Math.PI);
            landmarksCtx.fill();
          });
          connectLandmarks(landmarksCtx, data.landmarks);
        }
      }

      window.onload = function () {
        showTab("upload");
      };

      window.onbeforeunload = function () {
        if (isStreaming) stopWebcam();
        if (socket) socket.disconnect();
      };
    </script>
  </body>
</html>
