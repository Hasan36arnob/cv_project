from flask import Flask, render_template, request
import cv2, os, torch
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ✅ Load YOLO (Ultralytics)
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt")  # ছোট, দ্রুত মডেল

@app.route("/", methods=["GET", "POST"])
def index():
    uploaded, yolo_out = None, None  # Removed clip_text

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            uploaded = filepath

            # -------- YOLO Inference ----------
            results = yolo_model(filepath)
            results[0].save(filename="static/outputs/yolo_" + filename)
            yolo_out = "static/outputs/yolo_" + filename

    return render_template("index.html", uploaded=uploaded, yolo_out=yolo_out)  # Removed clip_text

if __name__ == "__main__":
    app.run(debug=True)
