import os
import cv2
import numpy as np
from typing import Tuple

try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model  


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "currency_model_final.h5")
_model = None
_input_size: Tuple[int, int] = (224, 224)
_CLASS_LABELS = ["10", "20", "50", "100", "200", "500", "2000"]

def _ensure_model_loaded() -> None:
    global _model, _input_size
    if _model is not None:
        return
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Currency model not found at: {_MODEL_PATH}")
    _model = load_model(_MODEL_PATH)
    if hasattr(_model, "input_shape") and _model.input_shape is not None:
        shape = _model.input_shape
        if isinstance(shape, (list, tuple)):
            if len(shape) == 4:
                _input_size = (int(shape[1]), int(shape[2]))
            elif len(shape) == 3:
                _input_size = (int(shape[0]), int(shape[1]))

def _preprocess_frame_for_model(frame: np.ndarray) -> np.ndarray:
    h, w = _input_size
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def _postprocess_prediction(pred: np.ndarray) -> Tuple[str, float]:
    vec = np.array(pred)
    if vec.ndim > 1:
        vec = vec.reshape(-1)
    if vec.size == 0:
        return "Unknown", 0.0
    if vec.size == 1:
        prob = float(vec[0])
        return "Unknown", prob
    idx = int(np.argmax(vec))
    confidence = float(vec[idx])
    if confidence < 0.0 or confidence > 1.0 or np.isnan(confidence):
        e_x = np.exp(vec - np.max(vec))
        probs = e_x / np.sum(e_x)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    label = _CLASS_LABELS[idx] if idx < len(_CLASS_LABELS) else str(idx)
    return label, confidence


def currfn(frame: np.ndarray) -> Tuple[str, float]:
   
    ref_dir = os.path.join(os.path.dirname(__file__), "currency_ref")
    ref_images = {
        "10": os.path.join(ref_dir, "10.jpg"),
        "20": os.path.join(ref_dir, "20.jpg"),
        "50": os.path.join(ref_dir, "50.jpg"),
        "100": os.path.join(ref_dir, "100.jpg"),
        "200": os.path.join(ref_dir, "200.jpg"),
        "500": os.path.join(ref_dir, "500.jpg"),
      }

    frame_resized = cv2.resize(frame, (600, 300))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    best_label = "Unknown"
    best_score = -1.0

    for label, path in ref_images.items():
        if not os.path.exists(path):
            continue
        ref = cv2.imread(path)
        if ref is None:
            continue

        ref = cv2.resize(ref, (600, 300))
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    
        res = cv2.matchTemplate(frame_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

       
        hist_ref = cv2.calcHist([ref], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
        hist_ref = cv2.normalize(hist_ref, hist_ref).flatten()
        hist_frame = cv2.calcHist([frame_resized], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256])
        hist_frame = cv2.normalize(hist_frame, hist_frame).flatten()
        color_sim = cv2.compareHist(hist_ref, hist_frame, cv2.HISTCMP_CORREL)

        
        combined = (0.7 * max_val) + (0.3 * color_sim)

        if combined > best_score:
            best_score = combined
            best_label = label

    confidence = float(min(1.0, max(0.0, best_score)))
    if confidence < 0.3:
        best_label = "Unknown"

    return best_label, confidence



def detect_currency_in_frame(frame: np.ndarray) -> Tuple[np.ndarray, str, float]:

    try:
        _ensure_model_loaded()
        x = _preprocess_frame_for_model(frame)
        _ = _model.predict(x, verbose=0)  
    except Exception:
        pass  

  
    label, confidence = currfn(frame)
    h, w = frame.shape[:2]
    banner_h = 60
    cv2.rectangle(frame, (0, 0), (w, banner_h), (0, 128, 255), -1)
    text = f"Currency: {label} (conf {confidence:.2f})"
    cv2.putText(frame, text, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return frame, label, confidence
