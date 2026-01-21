from ultralytics import YOLO
import cv2

model = YOLO('yolov8l.pt')

def detect_objects_in_frame(frame):
    results = model(frame, verbose=False)
    result = results[0]

    detected_objects = []

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        if confidence > 0.5:
            detected_objects.append(class_name)

            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_objects
