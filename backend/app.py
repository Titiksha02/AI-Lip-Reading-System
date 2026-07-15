import os
import subprocess

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory
)

from predict import predict_video

app = Flask(
    __name__,
    static_folder="../frontend",
    static_url_path=""
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return send_from_directory(
        "../frontend",
        "index.html"
    )


# -----------------------------
# Serve uploaded videos
# -----------------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:
        return jsonify({
            "prediction": "No video uploaded"
        })

    file = request.files["video"]

    # Save uploaded video (.mpg)
    input_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(input_path)

    filename = os.path.splitext(file.filename)[0]

    # Output mp4 path
    output_path = os.path.join(
        UPLOAD_FOLDER,
        filename + ".mp4"
    )

    print("Converting video...")

    # Convert MPG → MP4
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            output_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print("Conversion Finished")

    # Predict using original video
    prediction = predict_video(input_path)

    return jsonify({
        "prediction": prediction,
        "video": "/uploads/" + filename + ".mp4"
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )