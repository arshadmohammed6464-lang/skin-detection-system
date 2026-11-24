import os
import flask
import joblib
import numpy as np
from PIL import Image

app = flask.Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = os.path.join(BASE_DIR, "skin_model.pkl")
model = joblib.load(MODEL_PATH)


def prepare_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Agar model 2D input (samples, features) chahta hai
    if getattr(model, "ndim", None) == 2 or hasattr(model, "n_features_in_"):
        expected = getattr(model, "n_features_in_", arr.size)
        flat = arr.flatten()
        if flat.size != expected:
            side = int(round((expected / 3) ** 0.5))
            if side > 0 and side * side * 3 == expected:
                img = Image.open(image_path).convert("RGB").resize((side, side))
                flat = (np.array(img, dtype=np.float32) / 255.0).flatten()
        return flat.reshape(1, -1)

    return np.expand_dims(arr, axis=0)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if flask.request.method == "POST":
        file = flask.request.files.get("file")
        if not file or file.filename == "":
            prediction = "Please select an image file first."
            return flask.render_template(
                "index.html", prediction=prediction, confidence=confidence
            )

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        try:
            features = prepare_image(save_path)
            pred_result = model.predict(features)[0]
            prediction = str(pred_result)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                confidence = f"{np.max(probs) * 100:.2f}%"
            else:
                confidence = "N/A"
        except Exception as e:
            prediction = f"Error: {str(e)}"
            confidence = "N/A"

    return flask.render_template(
        "index.html", prediction=prediction, confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
