from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64, numpy as np, cv2, os, logging

app = Flask(__name__)
CORS(app, origins=["http://localhost", "http://localhost:80", "http://127.0.0.1", "http://localhost:8000"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_FOLDER = "dataset"
TEMP_FOLDER = "temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

MODEL_NAME = "Facenet512"
DISTANCE_METRIC = "cosine"
MATCH_THRESHOLD = 0.25


def decode_base64_image(b64_string: str, save_path: str) -> bool:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_bytes = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("cv2.imdecode returned None — invalid image bytes")
            return False
        cv2.imwrite(save_path, img)
        logger.info(f"Saved image to {save_path} ({os.path.getsize(save_path)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return False


def has_face(img_path: str) -> bool:
    try:
        faces = DeepFace.extract_faces(
            img_path=img_path,
            enforce_detection=False,
            silent=True,
        )
        valid = [f for f in faces if f.get("confidence", 0) > 0.5]
        logger.info(f"Faces found: {len(faces)}, valid (conf>0.5): {len(valid)}")
        return len(valid) > 0
    except Exception as e:
        logger.warning(f"Face extraction error (will proceed anyway): {e}")
        return True


def verify_face(live_path: str, dataset_path: str) -> dict:
    try:
        result = DeepFace.verify(
            img1_path=live_path,
            img2_path=dataset_path,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            silent=True,
        )
        distance = round(result["distance"], 4)
        MAX_DIST = 0.6
        confidence = round(max(0, (1 - distance / MAX_DIST) * 100), 1)
        logger.info(f"  {os.path.basename(dataset_path)} -> distance={distance}, match={result['verified']}")
        return {
            "name": os.path.splitext(os.path.basename(dataset_path))[0],
            "filename": os.path.basename(dataset_path),
            "match": distance <= MATCH_THRESHOLD,
            "distance": distance,
            "confidence": min(confidence, 100.0),
        }
    except Exception as e:
        logger.warning(f"Verify failed for {dataset_path}: {e}")
        return {
            "name": os.path.splitext(os.path.basename(dataset_path))[0],
            "filename": os.path.basename(dataset_path),
            "match": False,
            "distance": None,
            "confidence": 0.0,
        }


@app.route("/health", methods=["GET"])
def health():
    dataset_count = len([
        f for f in os.listdir(DATASET_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return jsonify({"status": "ok", "dataset_count": dataset_count})


@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing image field"}), 400

    live_path = os.path.join(TEMP_FOLDER, "live.jpg")

    if not decode_base64_image(data["image"], live_path):
        return jsonify({"error": "Invalid image data — could not decode."}), 400

    if not has_face(live_path):
        return jsonify({
            "status": "no_face",
            "message": "No face detected. Please upload a clear, well-lit photo facing the camera."
        })

    dataset_files = [
        os.path.join(DATASET_FOLDER, f)
        for f in os.listdir(DATASET_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not dataset_files:
        return jsonify({"status": "error", "message": "Dataset folder is empty."}), 500

    logger.info(f"Comparing against {len(dataset_files)} dataset image(s)...")

    results = [verify_face(live_path, path) for path in dataset_files]
    matched = [r for r in results if r["match"]]

    if matched:
        best = max(matched, key=lambda x: x["confidence"])
        return jsonify({
            "status": "matched",
            "matched_with": best["name"],
            "filename": best["filename"],
            "distance": best["distance"],
            "confidence": best["confidence"],
            "all_matches": len(matched),
        })
    else:
        valid_results = [r for r in results if r["distance"] is not None]
        closest = min(valid_results, key=lambda x: x["distance"], default=None)
        return jsonify({
            "status": "no_match",
            "closest": closest["name"] if closest else None,
            "closest_confidence": closest["confidence"] if closest else 0,
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)