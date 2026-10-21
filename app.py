
from pathlib import Path
import uuid
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from flask import Flask, render_template, request, make_response
import numpy as np
import tensorflow as tf
from database import initialize_database, save_prediction, get_history
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "image_model_mobilenet_finetune_v2.keras"
)

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

IMAGE_SIZE = (160, 160)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
initialize_database()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Load the trained CNN model
model = None
if MODEL_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS



def predict_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(img, dtype=np.float32)

    # Model already contains a Rescaling layer.
    # Pass raw RGB pixel values in the range 0–255.
    image_array = np.expand_dims(image_array, axis=0)

    score = float(model.predict(image_array, verbose=0)[0][0])

    # Model class mapping: Fake=0, Real=1
    if score >= 0.5:
        prediction = "Real"
        confidence = score * 100
    else:
        prediction = "Fake"
        confidence = (1 - score) * 100

    return prediction, confidence



@app.route("/", methods=["GET"])
def home():
    history_records = get_history(limit=100000)

    total_images = len(history_records)

    real_count = sum(
        1 for item in history_records
        if item["prediction"] == "Real"
    )

    fake_count = sum(
        1 for item in history_records
        if item["prediction"] == "Fake"
    )

    return render_template(
        "index.html",
        total_images=total_images,
        real_count=real_count,
        fake_count=fake_count
    )



@app.route("/history", methods=["GET"])
def history_page():
    history_records = get_history()
    return render_template(
        "history.html",
        history=history_records,
    )


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    history_records = get_history(limit=100000)

    total_images = len(history_records)
    real_count = sum(
        1 for item in history_records
        if item["prediction"] == "Real"
    )
    fake_count = sum(
        1 for item in history_records
        if item["prediction"] == "Fake"
    )

    return render_template(
        "dashboard.html",
        total_images=total_images,
        real_count=real_count,
        fake_count=fake_count,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            "Trained model not found. Please check models/image_model.keras.",
            500,
        )

    if "image" not in request.files:
        return "Please select an image.", 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return "No image selected.", 400

    if not allowed_file(uploaded_file.filename):
        return "Unsupported file type. Please upload an image.", 400

    # Generate a unique filename to avoid overwriting uploads
    safe_name = secure_filename(uploaded_file.filename)
    extension = Path(safe_name).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    image_path = UPLOAD_FOLDER / unique_name

    try:
        uploaded_file.save(image_path)

        # Verify that the uploaded file is a readable image
        with Image.open(image_path) as img:
            img.verify()

        prediction, confidence = predict_image(image_path)
        save_prediction(unique_name, prediction, round(confidence, 2))

    except (OSError, UnidentifiedImageError, ValueError):
        image_path.unlink(missing_ok=True)
        return "Could not read this image. Please upload a valid image.", 400

    return render_template(
        "result.html",
        filename=unique_name,
        prediction=prediction,
        confidence=round(confidence, 2),
    )


@app.errorhandler(413)
def file_too_large(error):
    return "Image is too large. Maximum upload size is 10 MB.", 413


@app.route("/download-report", methods=["GET"])
def download_report():
    records = get_history(limit=100000)

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    page_width, page_height = A4
    y = page_height - 50

    pdf.setTitle("DeepShield AI Prediction Report")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "DeepShield AI")
    y -= 30

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, "Image Prediction History Report")
    y -= 30

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Total records: {len(records)}")
    y -= 30

    for item in records:
        if y < 80:
            pdf.showPage()
            y = page_height - 50
            pdf.setFont("Helvetica", 10)

        pdf.drawString(50, y, f"ID: {item['id']}")
        y -= 15

        pdf.drawString(
            50, y,
            f"File: {item['filename'][:55]}"
        )
        y -= 15

        pdf.drawString(
            50, y,
            f"Prediction: {item['prediction']} | "
            f"Confidence score: {item['confidence']}%"
        )
        y -= 15

        pdf.drawString(
            50, y,
            f"Date & Time (UTC): {item['created_at']}"
        )
        y -= 25

    pdf.save()
    buffer.seek(0)

    response = make_response(buffer.getvalue())
    response.headers["Content-Type"] = "application/pdf"
    response.headers[
        "Content-Disposition"
    ] = "attachment; filename=DeepShield_AI_Report.pdf"

    return response


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
