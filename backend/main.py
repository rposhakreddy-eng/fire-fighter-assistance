"""
FastAPI Backend for Fire Fighter Assistance Drone System
Provides real AI inference pipeline, webcam/drone live feed processing,
SQLite persistent detection history, thermal colormaps, and emergency alerts.
"""

import os
import time
import uuid
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from backend.detector import detector_instance
from backend.grok_ai import analyze_with_grok, is_groq_configured
from backend.drone_stream import drone_manager
import backend.database as db

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Fire Fighter Assistance Drone API",
    description="Real AI Detection & Thermal Analysis API for Firefighter Search & Rescue",
    version="2.0.0"
)

# CORS Middleware for local frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static results directory so annotated images and reports can be fetched
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Mount frontend directory for easy single-port access
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# Pydantic Request Models
class DetectionUpdate(BaseModel):
    status: str
    notes: Optional[str] = ""


class AlertCreate(BaseModel):
    detection_id: str
    analysis_id: str
    victim_status: str
    confidence: float
    location_summary: str
    notes: Optional[str] = ""


class DroneConfigUpdate(BaseModel):
    url: str
    protocol: Optional[str] = "RTSP"


class FrameData(BaseModel):
    image_base64: str
    confidence_threshold: Optional[float] = 0.25


class LocationUpdateRequest(BaseModel):
    analysis_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    floor: str
    room: str
    status: Optional[str] = "Operator Verified Location"



@app.get("/")
def home():
    """Root endpoint with system status overview."""
    return {
        "system": "Fire Fighter Assistance Drone AI Backend",
        "status": "online",
        "version": "2.0.0",
        "frontend_url": "/app/index.html",
        "docs_url": "/docs",
        "models": detector_instance.get_status()
    }


@app.get("/health")
def health():
    """Comprehensive health check for frontend status indicators."""
    model_status = detector_instance.get_status()
    db_stats = db.get_system_counts()
    return {
        "status": "healthy",
        "backend": "online",
        "models": {
            "detection_ready": model_status["detection_model_loaded"],
            "pose_ready": model_status["pose_model_loaded"],
            "dedicated_body_part_model": False,
            "body_part_status": "Dedicated body-part model not configured; pose keypoints mapped",
            "thermal_validation": "Thermal-image detection has not been validated for this model."
        },
        "database": "connected",
        "groq_ai": {
            "configured": is_groq_configured(),
            "status": "Ready (API Key Present)" if is_groq_configured() else "Not configured (Add GROQ_API_KEY to .env)"
        },
        "drone_stream": {
            "configured": bool(drone_manager.stream_url),
            "connected": drone_manager.is_connected,
            "status": "Connected" if drone_manager.is_connected else "Drone stream not configured"
        },
        "stats": db_stats,
        "timestamp": time.time()
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    source_type: str = Form("image"),
    palette: str = Form("normal"),
    gps_lat: Optional[float] = Form(None),
    gps_lon: Optional[float] = Form(None),
    floor: str = Form("Exact indoor location unavailable"),
    room: str = Form("Exact indoor location unavailable")
):
    """
    Primary image analysis endpoint.
    Performs real YOLO detection and pose estimation, Groq multimodal analysis,
    saves annotated image to disk, and records history into SQLite.
    """
    # 1. Validate file format
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")):
        raise HTTPException(status_code=400, detail="Unsupported image format. Please upload JPEG, PNG, or WebP.")

    session_id = f"ANL-{uuid.uuid4().hex[:8].upper()}"
    original_filename = f"orig_{session_id}_{file.filename}"
    annotated_filename = f"annotated_{session_id}.jpg"

    original_save_path = RESULTS_DIR / original_filename
    annotated_save_path = RESULTS_DIR / annotated_filename

    # 2. Save original file
    try:
        content = await file.read()
        if len(content) > 30 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 30MB limit.")

        with open(original_save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded image: {e}")

    # 3. Apply optional pseudo-thermal colormap if requested
    inference_image_path = str(original_save_path)
    if palette.lower() != "normal":
        colormap_img = detector_instance.apply_colormap(inference_image_path, palette)
        if colormap_img is not None:
            pseudo_filename = f"palette_{palette}_{session_id}.jpg"
            pseudo_path = RESULTS_DIR / pseudo_filename
            cv2.imwrite(str(pseudo_path), colormap_img)
            inference_image_path = str(pseudo_path)

    # 4. Perform Real AI Inference (YOLO11 Detection + Pose)
    try:
        det_result = detector_instance.detect_and_annotate(
            image_input=inference_image_path,
            conf_threshold=confidence_threshold,
            use_pose=True,
            save_annotated_path=str(annotated_save_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model inference failed: {e}")

    # 5. Run Groq Secondary Multimodal Vision Analysis
    grok_result = analyze_with_grok(
        image_path=str(original_save_path),
        detections=det_result["detections"]
    )

    # 6. Determine Location Status
    loc_status = "Exact indoor location unavailable"
    if gps_lat is not None and gps_lon is not None:
        loc_status = f"Drone GPS coordinates recorded ({gps_lat:.4f}, {gps_lon:.4f})"
    elif floor != "Exact indoor location unavailable" or room != "Exact indoor location unavailable":
        loc_status = "Manual verification attached"

    visible_body_parts_str = ", ".join(det_result["visible_body_parts"]) if det_result["visible_body_parts"] else "None identified"

    # 7. Persist to SQLite Database
    db.save_analysis(
        analysis_id=session_id,
        source_type=source_type,
        filename=file.filename,
        original_image_path=f"/results/{original_filename}",
        annotated_image_path=f"/results/{annotated_filename}",
        people_detected=det_result["people_count"],
        objects_summary=det_result["objects_summary"],
        best_confidence=det_result["best_confidence"],
        visible_body_parts=visible_body_parts_str,
        model_used=det_result["model_used"],
        thermal_palette=palette.upper(),
        gps_latitude=gps_lat,
        gps_longitude=gps_lon,
        floor_number=floor,
        room_number=room,
        location_status=loc_status,
        grok_assessment=grok_result.get("human_assessment", ""),
        grok_body_part=grok_result.get("visible_body_part", ""),
        grok_evidence=grok_result.get("evidence", ""),
        grok_recommendation=grok_result.get("recommendation", ""),
        status="completed"
    )

    # Persist individual detections
    for d in det_result["detections"]:
        db.save_detection(
            detection_id=f"{session_id}-{d['detection_id']}",
            analysis_id=session_id,
            class_name=d["class_name"],
            confidence=d["confidence"],
            bbox=d["bbox"],
            keypoints=d["keypoints"],
            visible_body_parts=", ".join(d["visible_body_parts"]),
            operator_status="Pending Review"
        )

    # Return structured response matching requirements
    return {
        "analysis_id": session_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file.filename,
        "source_type": source_type,
        "people_detected": det_result["people_count"],
        "best_confidence": det_result["best_confidence"],
        "objects_summary": det_result["objects_summary"],
        "detections": det_result["detections"],
        "visible_body_parts": det_result["visible_body_parts"],
        "body_part_status": "Dedicated body-part model not configured; pose keypoints mapped",
        "thermal_disclaimer": "Thermal-image detection has not been validated for this model. Temperature values are not measured without calibrated radiometric thermal sensors.",
        "model_used": det_result["model_used"],
        "inference_time_ms": det_result["inference_time_ms"],
        "original_image_url": f"/results/{original_filename}",
        "annotated_image_url": f"/results/{annotated_filename}",
        "location": {
            "latitude": gps_lat,
            "longitude": gps_lon,
            "floor": floor,
            "room": room,
            "status": loc_status,
            "indoor_positioning": "Indoor positioning & building map integration not configured"
        },
        "secondary_ai_analysis": grok_result
    }


@app.post("/analyze-frame")
async def analyze_frame(data: FrameData):
    """
    Real-time inference endpoint for live laptop webcam or streaming frames.
    Receives base64 image, executes model inference, and returns annotated frame.
    """
    b64_str = data.image_base64
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid frame image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decoding error: {e}")

    # Process frame
    res = drone_manager.process_live_frame(frame, conf_threshold=data.confidence_threshold or 0.25)
    return res


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    sample_interval: int = Form(5)
):
    """Process uploaded video file with frame sampling and tracking."""
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    video_id = f"VID-{uuid.uuid4().hex[:8].upper()}"
    video_path = RESULTS_DIR / f"{video_id}_{file.filename}"

    try:
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    try:
        video_result = drone_manager.process_video_file(
            video_path=str(video_path),
            output_dir=str(RESULTS_DIR),
            conf_threshold=confidence_threshold,
            sample_interval=max(1, sample_interval)
        )
        video_result["annotated_video_url"] = f"/results/{video_result['annotated_video_filename']}"
        return video_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {e}")


@app.get("/detections")
def get_detections(limit: int = 100):
    """Retrieve all detections from persistent SQLite database."""
    return db.get_all_detections(limit=limit)


@app.patch("/detections/{detection_id}")
def update_detection(detection_id: str, update: DetectionUpdate):
    """Update human verification status for a detection."""
    valid_statuses = ["Pending Review", "Possible Victim", "Confirmed Victim", "False Positive"]
    if update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

    success = db.update_detection_status(detection_id, update.status, update.notes or "")
    if not success:
        raise HTTPException(status_code=404, detail="Detection ID not found.")
    return {"message": "Detection updated successfully", "detection_id": detection_id, "status": update.status}


@app.get("/history")
def get_history(limit: int = 50):
    """Retrieve analysis session history."""
    return db.get_all_analyses(limit=limit)


@app.get("/history/{analysis_id}")
def get_history_detail(analysis_id: str):
    """Retrieve single analysis with its detections."""
    analysis = db.get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")
    return analysis


@app.delete("/history/{analysis_id}")
def delete_history_item(analysis_id: str):
    """Delete an analysis record and its detections."""
    success = db.delete_analysis(analysis_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")
    return {"message": "Analysis deleted successfully", "analysis_id": analysis_id}


@app.get("/alerts")
def get_alerts(limit: int = 50):
    """Retrieve emergency rescue alerts."""
    return db.get_all_alerts(limit=limit)


@app.post("/alerts")
def create_alert(alert: AlertCreate):
    """Prepare and record an emergency rescue alert for firefighters."""
    alert_id = f"ALT-{uuid.uuid4().hex[:6].upper()}"
    success = db.create_emergency_alert(
        alert_id=alert_id,
        detection_id=alert.detection_id,
        analysis_id=alert.analysis_id,
        victim_status=alert.victim_status,
        confidence=alert.confidence,
        location_summary=alert.location_summary,
        notes=alert.notes or ""
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to record emergency alert.")
    return {
        "alert_id": alert_id,
        "message": "Emergency alert recorded locally.",
        "dispatch_status": "Prepared (Notification service not configured)",
        "notification_note": "External SMS / dispatch integration not configured. Local alert created for responder review."
    }


@app.get("/drone-status")
def get_drone_status():
    """Retrieve drone connection status and telemetry."""
    test_result = drone_manager.test_connection()
    return {
        "stream_url": drone_manager.stream_url or "Not configured",
        "protocol": drone_manager.stream_protocol,
        "connection": test_result,
        "telemetry": drone_manager.drone_telemetry
    }


@app.post("/drone-config")
def update_drone_config(config: DroneConfigUpdate):
    """Update drone stream URL and test connection."""
    drone_manager.set_config(url=config.url, protocol=config.protocol or "RTSP")
    test_result = drone_manager.test_connection()
    return {
        "message": "Drone configuration updated",
        "stream_url": drone_manager.stream_url,
        "connection": test_result
    }


@app.get("/locations/markers")
def get_location_markers():
    """Retrieve all victim locations with coordinates and floor details for map display."""
    markers = db.get_all_victim_map_markers()
    return {
        "count": len(markers),
        "markers": markers
    }


@app.post("/locations/update")
def update_victim_location(req: LocationUpdateRequest):
    """Update exact coordinates, floor, or room where camera captured the victim."""
    success = db.update_analysis_location(
        analysis_id=req.analysis_id,
        lat=req.latitude,
        lon=req.longitude,
        floor=req.floor,
        room=req.room,
        status=req.status or "Operator Verified Location"
    )
    if not success:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")
    return {
        "message": "Location updated successfully",
        "analysis_id": req.analysis_id,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "floor": req.floor,
        "room": req.room
    }



@app.websocket("/ws/detect")
@app.websocket("/ws/drone-feed")
async def websocket_drone_feed(websocket: WebSocket):
    """Live WebSocket stream for drone/webcam frames."""
    await websocket.accept()
    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                payload = json.loads(text_data)
                b64_str = payload.get("image", payload.get("frame", payload.get("image_base64", "")))
                if b64_str:
                    if "," in b64_str:
                        b64_str = b64_str.split(",")[1]
                    img_bytes = base64.b64decode(b64_str)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        conf = float(payload.get("confidence", 0.25))
                        res = drone_manager.process_live_frame(frame, conf_threshold=conf)
                        await websocket.send_json(res)
                        continue
            except Exception as parse_err:
                pass
            await websocket.send_json({"status": "acknowledged", "timestamp": time.time()})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Client disconnected: {e}")