"""
Drone Stream and Video Processing Adapter
Handles live drone feeds (RTSP/HTTP), webcam frame analysis,
and uploaded video processing with frame sampling and tracking.
"""

import os
import cv2
import time
import base64
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.detector import detector_instance


class DroneStreamManager:
    """Manages drone stream configurations and video processing."""

    def __init__(self):
        self.stream_url = ""
        self.stream_protocol = "RTSP"  # RTSP, HTTP, UDP
        self.is_connected = False
        self.last_frame = None
        self.last_check_time = None
        self.drone_telemetry = {
            "altitude_m": "N/A",
            "battery_pct": "N/A",
            "gps_lat": None,
            "gps_lon": None,
            "signal_quality": "Not connected"
        }

    def set_config(self, url: str, protocol: str = "RTSP"):
        self.stream_url = url.strip()
        self.stream_protocol = protocol.upper()
        self.is_connected = False

    def test_connection(self) -> Dict[str, Any]:
        """Test whether configured drone stream is reachable."""
        if not self.stream_url:
            return {
                "status": "not_configured",
                "connected": False,
                "message": "Drone camera stream URL is not configured in Settings."
            }

        try:
            # Attempt to open video capture with short timeout
            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)

            if not cap.isOpened():
                cap.release()
                self.is_connected = False
                return {
                    "status": "unreachable",
                    "connected": False,
                    "message": f"Unable to connect to stream at {self.stream_url}. Verify network and drone power."
                }

            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                self.is_connected = True
                self.last_frame = frame
                self.last_check_time = time.time()
                return {
                    "status": "connected",
                    "connected": True,
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                    "message": "Drone stream connection established successfully."
                }
            else:
                self.is_connected = False
                return {
                    "status": "no_signal",
                    "connected": False,
                    "message": "Stream connected but returned no video frames."
                }

        except Exception as e:
            self.is_connected = False
            return {
                "status": "error",
                "connected": False,
                "message": f"Stream connection error: {str(e)}"
            }

    def process_live_frame(self, frame_bgr: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Process a single frame from webcam or drone feed.
        Returns detection metadata and JPEG base64 annotated frame for instant client display.
        """
        t0 = time.time()
        result = detector_instance.detect_and_annotate(
            frame_bgr,
            conf_threshold=conf_threshold,
            use_pose=True,
            save_annotated_path=None
        )

        # Encode annotated image to base64 JPEG
        annotated_img = result.pop("annotated_image")
        success, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_frame = ""
        if success:
            b64_frame = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        result["annotated_frame_base64"] = b64_frame
        result["fps"] = round(1.0 / max(0.001, (time.time() - t0)), 1)
        return result

    def process_video_file(
        self,
        video_path: str,
        output_dir: str,
        conf_threshold: float = 0.30,
        sample_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Process an uploaded video file, analyzing every `sample_interval` frame.
        Generates annotated output video and frame-by-frame detection timeline.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(video_path).stem
        out_video_path = os.path.join(output_dir, f"annotated_{base_name}.mp4")

        # FourCC codec for mp4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_video_path, fourcc, orig_fps / sample_interval, (width, height))

        frame_idx = 0
        analyzed_frames = 0
        max_people_in_single_frame = 0
        timeline_events = []
        keyframe_images = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                analyzed_frames += 1
                det_res = detector_instance.detect_and_annotate(
                    frame,
                    conf_threshold=conf_threshold,
                    use_pose=True,
                    save_annotated_path=None
                )
                annotated = det_res["annotated_image"]
                out_writer.write(annotated)

                people = det_res["people_count"]
                if people > max_people_in_single_frame:
                    max_people_in_single_frame = people

                if people > 0:
                    timestamp_sec = round(frame_idx / orig_fps, 2)
                    timeline_events.append({
                        "frame_index": frame_idx,
                        "timestamp_seconds": timestamp_sec,
                        "people_detected": people,
                        "best_confidence": det_res["best_confidence"],
                        "visible_body_parts": det_res["visible_body_parts"]
                    })

                    # Save first 3 keyframes
                    if len(keyframe_images) < 3:
                        kf_name = f"keyframe_{base_name}_{frame_idx}.jpg"
                        kf_path = os.path.join(output_dir, kf_name)
                        cv2.imwrite(kf_path, annotated)
                        keyframe_images.append(kf_name)

            frame_idx += 1

        cap.release()
        out_writer.release()

        return {
            "total_video_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "duration_seconds": round(total_frames / orig_fps, 1),
            "max_people_in_single_frame": max_people_in_single_frame,
            "detection_events_count": len(timeline_events),
            "timeline": timeline_events[:50],  # cap for response size
            "keyframes": keyframe_images,
            "annotated_video_filename": os.path.basename(out_video_path)
        }


# Global manager instance
drone_manager = DroneStreamManager()
