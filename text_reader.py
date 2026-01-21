import cv2
import numpy as np
from paddleocr import PaddleOCR
import pyttsx3
import threading

# Initialize PaddleOCR and TTS
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # or 'en+hi' if you want Hindi support
engine = pyttsx3.init()
lock = threading.Lock()  # prevents overlapping speech calls

def say(text):
    """Thread-safe TTS function"""
    print(f"TTS: {text}")
    def speak():
        with lock:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS ERROR] {e}")
    threading.Thread(target=speak, daemon=True).start()

def read_text_from_scene(frame):
    """
    Detects and reads text from a frame using PaddleOCR.
    Draws boxes, speaks text aloud, and returns annotated frame.
    """
    try:
        print("[INFO] Detecting text with PaddleOCR...")
        results = ocr.ocr(frame, cls=True)

        if not results or not results[0]:
            say("No readable text found.")
            cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)
            cv2.putText(frame, "No text detected", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return frame, "No text detected"

        detected_texts = []
        for line in results[0]:
            box, (text, confidence) = line
            if confidence < 0.4:  # skip weak detections
                continue
            detected_texts.append(text)

            # Draw bounding boxes and text on frame
            pts = np.array(box).astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, text, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined_text = " ".join(detected_texts)
        print(f"[INFO] Extracted Text: {combined_text}")
        say(f"I found the following text: {combined_text}")

        return frame, combined_text

    except Exception as e:
        print(f"[ERROR] PaddleOCR failed: {e}")
        say("Sorry, I could not read any text.")
        return frame, "Error reading text"
