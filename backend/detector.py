from ultralytics import YOLO


# Load YOLO model
model = YOLO("../models/yolo11n.pt")


def detect_image(image_path):

    results = model(
        image_path,
        conf=0.25
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            detections.append({
                "object": object_name,
                "confidence": confidence
            })

    return detections