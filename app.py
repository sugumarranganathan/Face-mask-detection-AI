from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load Face Detector
faceNet = cv2.dnn.readNet(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Load Mask Model
maskNet = load_model("mask_detector.h5", compile=False)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Mask Detection AI</title>
</head>
<body>

<h1>Face Mask Detection AI</h1>

<form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" required>
    <button type="submit">Detect</button>
</form>

<br>

{% if image %}
<img src="data:image/jpeg;base64,{{ image }}" width="600">
{% endif %}

</body>
</html>
"""

def detect_and_predict_mask(frame):

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face = cv2.resize(
                face,
                (224, 224)
            )

            face = img_to_array(face)

            face = preprocess_input(face)

            face = np.expand_dims(
                face,
                axis=0
            )

            (mask, withoutMask) = maskNet.predict(
                face,
                verbose=0
            )[0]

            if mask > withoutMask:

                label = "Mask Detected"
                color = (0, 255, 0)

            else:

                label = "No Mask Detected"
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    return frame

@app.route("/", methods=["GET", "POST"])
def home():

    image_base64 = None

    if request.method == "POST":

        file = request.files["image"]

        image = Image.open(file.stream).convert("RGB")

        frame = np.array(image)

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        frame = detect_and_predict_mask(frame)

        _, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        image_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")

    return render_template_string(
        HTML,
        image=image_base64
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
