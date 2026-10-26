"""
Computer Vision & AI Detector Module
Handles object detection (YOLO11n) and human pose keypoint estimation (YOLO11n-pose),
bounding box rendering, visible body part identification, and thermal colormap simulation.
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO

# Absolute paths to models
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
YOLO_DET_PATH = MODELS_DIR / "yolo11n.pt"
YOLO_POSE_PATH = MODELS_DIR / "yolo11n-pose.pt"

# Inference resolution. Lower = faster on weak CPUs (e.g. Render's free 0.1 CPU plan).
# Must be a multiple of 32; defaults to YOLO's native 640.
IMG_SIZE = max(160, int(os.getenv("YOLO_IMG_SIZE", "640")))

# Live-frame inference is slower than a single uploaded image because frames keep
# arriving. Use a smaller default so one frame stays well inside Render's ~100 s
# proxy timeout on the free plan.
FRAME_IMG_SIZE = max(160, int(os.getenv("YOLO_FRAME_IMG_SIZE", str(min(IMG_SIZE, 256)))))

# Fallback path if pose is in backend root
if not YOLO_POSE_PATH.exists() and (BASE_DIR / "yolo11n-pose.pt").exists():
    YOLO_POSE_PATH = BASE_DIR / "yolo11n-pose.pt"

# COCO 17 Keypoint names
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Skeleton connections for drawing pose
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Facial connections
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Upper body & arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


class AIDetector:
    """Manages YOLO detection and pose estimation models."""

    def __init__(self):
        self.detect_model: Optional[YOLO] = None
        self.pose_model: Optional[YOLO] = None
        self.active_model_name = "yolo11n.pt"
        self.load_models()

    def load_models(self):
        """Safely load detection and pose models."""
        try:
            if YOLO_DET_PATH.exists():
                self.detect_model = YOLO(str(YOLO_DET_PATH))
                print(f"[Detector] Loaded detection model: {YOLO_DET_PATH}")
            else:
                print(f"[Detector] WARNING: Detection model not found at {YOLO_DET_PATH}")

            if YOLO_POSE_PATH.exists():
                self.pose_model = YOLO(str(YOLO_POSE_PATH))
                print(f"[Detector] Loaded pose model: {YOLO_POSE_PATH}")
            else:
                print(f"[Detector] Pose model not found at {YOLO_POSE_PATH}")
        except Exception as e:
            print(f"[Detector] Error initializing models: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Check current model availability and status."""
        return {
            "detection_model_loaded": self.detect_model is not None,
            "detection_model_name": "YOLO11n (COCO 80 classes)",
            "pose_model_loaded": self.pose_model is not None,
            "pose_model_name": "YOLO11n-pose (17 human keypoints)",
            "dedicated_body_part_model": False,
            "body_part_note": "Dedicated body-part model not configured; pose keypoints mapped",
            "thermal_validation_note": "Thermal-image detection has not been validated for this model."
        }

    def identify_visible_body_parts(self, keypoints_list: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze high-confidence pose keypoints to determine visible anatomical regions.
        Keypoints confidence threshold: 0.4
        """
        visible_parts = set()
        kpt_conf = {k["name"]: k["confidence"] for k in keypoints_list if k["confidence"] >= 0.4}

        # Head / Facial region
        head_kpts = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear"]
        if any(h in kpt_conf for h in head_kpts):
            visible_parts.add("Head/Face")

        # Upper Body / Shoulders
        if "Left Shoulder" in kpt_conf or "Right Shoulder" in kpt_conf:
            visible_parts.add("Shoulders/Torso")

        # Arms & Hands
        arms = ["Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist"]
        if any(a in kpt_conf for a in arms):
            visible_parts.add("Arms/Hands")

        # Legs & Feet
        legs = ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
        if any(l in kpt_conf for l in legs):
            visible_parts.add("Legs/Feet")

        return sorted(list(visible_parts))

    def detect_and_annotate(
        self,
        image_input,
        conf_threshold: float = 0.25,
        use_pose: bool = True,
        save_annotated_path: Optional[str] = None,
        imgsz: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run real inference on an image (filepath or numpy BGR array).
        Returns bounding boxes, confidence percentages, pose keypoints, and annotated image.
        """
        t0 = time.time()

        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to read image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise ValueError("image_input must be a file path string or numpy array")

        img_h, img_w = image.shape[:2]
        annotated_img = image.copy()

        detections = []
        people_count = 0
        objects_summary: Dict[str, int] = {}
        all_visible_body_parts: set = set()

        # Step 1: Run standard detection or pose estimation
        # If pose model is available, pose model gives both person bounding boxes AND 17 keypoints!
        chosen_model = self.pose_model if (use_pose and self.pose_model is not None) else self.detect_model

        if chosen_model is None:
            raise RuntimeError("No YOLO model is currently loaded in the detector.")

        results = chosen_model(image, conf=conf_threshold, imgsz=imgsz or IMG_SIZE, verbose=False)
        det_index = 1

        for result in results:
            boxes = result.boxes
            keypoints_data = getattr(result, "keypoints", None)

            if boxes is not None:
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    class_name = chosen_model.names[cls_id]
                    confidence = float(box.conf[0])
                    conf_pct = round(confidence * 100, 1)

                    # Bounding box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # Clamp to image dimensions
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

                    # Tally object counts
                    objects_summary[class_name] = objects_summary.get(class_name, 0) + 1
                    if class_name.lower() == "person":
                        people_count += 1

                    # Extract Keypoints if this is a person and keypoints exist
                    item_keypoints = []
                    item_body_parts = []

                    if keypoints_data is not None and idx < len(keypoints_data.data):
                        kpts = keypoints_data.data[idx].tolist()
                        for kpt_idx, kpt in enumerate(kpts):
                            if kpt_idx < len(KEYPOINT_NAMES):
                                kx, ky = float(kpt[0]), float(kpt[1])
                                kconf = float(kpt[2]) if len(kpt) > 2 else 0.0
                                item_keypoints.append({
                                    "id": kpt_idx,
                                    "name": KEYPOINT_NAMES[kpt_idx],
                                    "x": round(kx, 1),
                                    "y": round(ky, 1),
                                    "confidence": round(kconf, 2)
                                })
                        item_body_parts = self.identify_visible_body_parts(item_keypoints)
                        for bp in item_body_parts:
                            all_visible_body_parts.add(bp)

                    det_id = f"DET-{det_index:03d}"
                    det_index += 1

                    detections.append({
                        "detection_id": det_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "confidence_percentage": conf_pct,
                        "bbox": [x1, y1, x2, y2],
                        "keypoints": item_keypoints,
                        "visible_body_parts": item_body_parts
                    })

                    # Render Annotations on annotated_img
                    # Colors in BGR: Person -> Fire Red/Orange (45, 60, 240), Other -> Cyan/Blue (220, 150, 40)
                    is_person = class_name.lower() == "person"
                    color = (40, 50, 235) if is_person else (230, 160, 30)

                    # Bounding Box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

                    # Label badge
                    label_text = f"{det_id}: {class_name.capitalize()} {conf_pct}%"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

                    badge_y1 = max(0, y1 - txt_h - 8)
                    badge_y2 = y1
                    cv2.rectangle(annotated_img, (x1, badge_y1), (x1 + txt_w + 10, badge_y2), color, -1)
                    cv2.putText(
                        annotated_img,
                        label_text,
                        (x1 + 5, badge_y2 - 4),
                        font,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        cv2.LINE_AA
                    )

                    # Draw Pose Skeleton if available
                    if item_keypoints:
                        # Draw connections
                        kpt_map = {k["id"]: (int(k["x"]), int(k["y"]), k["confidence"]) for k in item_keypoints}
                        for p1_id, p2_id in SKELETON_PAIRS:
                            if p1_id in kpt_map and p2_id in kpt_map:
                                x_a, y_a, c_a = kpt_map[p1_id]
                                x_b, y_b, c_b = kpt_map[p2_id]
                                if c_a >= 0.4 and c_b >= 0.4:
                                    cv2.line(annotated_img, (x_a, y_a), (x_b, y_b), (50, 220, 120), 2, cv2.LINE_AA)

                        # Draw Keypoints dots
                        for kpt in item_keypoints:
                            if kpt["confidence"] >= 0.4:
                                cv2.circle(annotated_img, (int(kpt["x"]), int(kpt["y"])), 4, (0, 255, 255), -1, cv2.LINE_AA)

        inference_time_ms = round((time.time() - t0) * 1000, 1)

        # Save annotated image if requested
        if save_annotated_path:
            os.makedirs(os.path.dirname(save_annotated_path), exist_ok=True)
            cv2.imwrite(save_annotated_path, annotated_img)

        best_conf = max([d["confidence_percentage"] for d in detections], default=0.0)

        return {
            "detections": detections,
            "people_count": people_count,
            "objects_summary": objects_summary,
            "best_confidence": best_conf,
            "visible_body_parts": sorted(list(all_visible_body_parts)),
            "model_used": chosen_model.ckpt_path if hasattr(chosen_model, "ckpt_path") else "YOLO11",
            "inference_time_ms": inference_time_ms,
            "image_dimensions": {"width": img_w, "height": img_h},
            "annotated_image": annotated_img
        }

    @staticmethod
    def apply_colormap(image_path: str, palette: str) -> Optional[np.ndarray]:
        """
        Generate pseudo-thermal visualization using OpenCV colormaps.
        Supported palettes: IRONBOW (INFERNO), RAINBOW (JET), HOT, BONE (WHITE-HOT/BLACK-HOT)
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p = palette.upper()

        if "IRON" in p or "INFERNO" in p:
            return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        elif "RAINBOW" in p or "JET" in p:
            return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        elif "HOT" in p:
            return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif "BLACK-HOT" in p:
            inverted = cv2.bitwise_not(gray)
            return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        elif "WHITE-HOT" in p or "BONE" in p:
            return cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
        else:
            return img


# Global detector instance
detector_instance = AIDetector()