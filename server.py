from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS
import detector
import scene_describer
import currency_detector
from text_reader import read_text_from_scene

app = Flask(__name__, template_folder='templates')
CORS(app)

print("[INFO] Models loaded (BLIP for scene).")

def analyze_image(frame):
    """Run scene, text, and object analysis."""
    try:
        print("[INFO] Describing scene...")
        scene_description = scene_describer.generate_enhanced_caption(frame)

        print("[INFO] Reading text with PaddleOCR...")
        _, read_text = read_text_from_scene(frame)

        print("[INFO] Detecting objects...")
        _, object_list = detector.detect_objects_in_frame(frame)
        objects_found = ", ".join(object_list) if object_list else "No objects detected."

        return {
            "scene": scene_description,
            "text": read_text,
            "objects": objects_found
        }
    except Exception as e:
        print(f"[ERROR] analyze_image failed: {e}")
        return {
            "scene": "Error generating scene.",
            "text": "Error reading text.",
            "objects": "Error detecting objects."
        }


def analyze_currency(frame):
    """
    Detect currency denomination in the given frame.
    """
    try:
        annotated, label, confidence = currency_detector.detect_currency_in_frame(frame)
        label_out = str(label) if label else "Unknown"
        conf_out = float(confidence) if confidence else 0.0
        return {"currency": label_out, "confidence": round(conf_out, 4)}
    except Exception as e:
        print(f"[ERROR] Currency analyze failed: {e}")
        return {"currency": "Unknown", "confidence": 0.0}


@app.route('/')
def home():
    """
    Serves the main frontend page.
    Flask automatically finds index.html inside /templates.
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def handle_analysis():
    """
    Main API route for analyzing frames from the web app.
    Accepts an image and 'task' type ('currency' or 'all').
    """
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image file found"}), 400

    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        return jsonify({"error": f"Image decoding failed: {e}"}), 400

    task = request.form.get('task', 'all').lower()
    print(f"[INFO] Received task: {task}")

    if task == 'currency':
        results = analyze_currency(frame)
    else:
        results = analyze_image(frame)

    print(f"[INFO] Results: {results}")
    return jsonify(results)


if __name__ == "__main__":
    print("[INFO] Starting Flask server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
