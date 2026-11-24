import os
import flask
import joblib
import numpy as np
from PIL import Image

app = flask.Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = os.path.join(BASE_DIR, "skin_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("Model load error:", e)


def extract_10_features(image_path):
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    feats = np.array([
        np.mean(r), np.std(r),
        np.mean(g), np.std(g),
        np.mean(b), np.std(b),
        np.mean(arr), np.std(arr),
        np.max(arr), np.min(arr),
    ], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    expected = getattr(model, "n_features_in_", 10) if model is not None else 10

    if feats.size < expected:
        feats = np.pad(feats, (0, expected - feats.size))
    elif feats.size > expected:
        feats = feats[:expected]

    return feats.reshape(1, -1)


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

        if model is None:
            prediction = "Error: Model could not be loaded."
            return flask.render_template(
                "index.html", prediction=prediction, confidence=confidence
            )

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        try:
            features = extract_10_features(save_path)
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
