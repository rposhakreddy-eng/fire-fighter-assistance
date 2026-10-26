"""
Automated Test Suite for Fire Fighter Assistance Drone System
Tests backend health, YOLO inference pipeline, frame analysis,
database persistence, and error handling.
"""

import sys
import os
import json
import base64
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
import backend.database as db

client = TestClient(app)

def test_1_health_endpoint():
    print("\n--- Test 1: Health Endpoint ---")
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models"]["detection_ready"] is True
    assert data["models"]["pose_ready"] is True
    assert "Thermal-image detection has not been validated" in data["models"]["thermal_validation"]
    print("✓ Health check PASSED:", json.dumps(data, indent=2))

def test_2_real_yolo_inference_on_person():
    print("\n--- Test 2: Real YOLO Inference on Person Image ---")
    sample_image = PROJECT_ROOT / "images" / "sample_victim_person.jpg"
    assert sample_image.exists(), "Sample test image not found"

    with open(sample_image, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample_victim_person.jpg", f, "image/jpeg")},
            data={"confidence_threshold": 0.25, "palette": "normal", "floor": "2nd Floor", "room": "Corridor B"}
        )

    assert response.status_code == 200, f"Analysis failed: {response.text}"
    data = response.json()
    assert data["people_detected"] >= 1, "Expected at least 1 person detected"
    assert data["best_confidence"] > 50.0, "Expected confidence > 50%"
    assert len(data["detections"]) >= 1, "Expected detection items"
    det = data["detections"][0]
    assert "bbox" in det and len(det["bbox"]) == 4, "Expected 4 bbox coords"
    assert "keypoints" in det, "Expected keypoints in pose detection"
    assert "annotated_image_url" in data, "Expected annotated image path"
    print(f"✓ Person Detection PASSED: {data['people_detected']} person(s), Best Confidence: {data['best_confidence']}%, Body parts: {data['visible_body_parts']}")

def test_3_real_inference_on_thermal_hand():
    print("\n--- Test 3: Inference on Thermal Hand Image ---")
    sample_image = PROJECT_ROOT / "images" / "sample_thermal_hand.jpg"
    assert sample_image.exists(), "Sample thermal hand image not found"

    with open(sample_image, "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample_thermal_hand.jpg", f, "image/jpeg")},
            data={"confidence_threshold": 0.25, "palette": "ironbow"}
        )

    assert response.status_code == 200, f"Thermal analysis failed: {response.text}"
    data = response.json()
    assert "secondary_ai_analysis" in data
    print("✓ Thermal Analysis PASSED:", data["secondary_ai_analysis"])

def test_4_frame_analysis_for_webcam():
    print("\n--- Test 4: Live Frame Analysis (Webcam/Drone feed) ---")
    sample_image = PROJECT_ROOT / "images" / "sample_victim_person.jpg"
    with open(sample_image, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    response = client.post(
        "/analyze-frame",
        json={"image_base64": f"data:image/jpeg;base64,{b64_str}", "confidence_threshold": 0.25}
    )
    assert response.status_code == 200, f"Frame analysis failed: {response.text}"
    data = response.json()
    assert "annotated_frame_base64" in data
    assert "fps" in data
    assert data["people_count"] >= 1
    print(f"✓ Frame Analysis PASSED: FPS {data['fps']}, People {data['people_count']}")

def test_5_invalid_file_handling():
    print("\n--- Test 5: Invalid File Upload Handling ---")
    response = client.post(
        "/analyze",
        files={"file": ("dummy.txt", b"This is not an image", "text/plain")},
        data={"confidence_threshold": 0.25}
    )
    assert response.status_code == 400, f"Expected 400 for text file, got {response.status_code}"
    print("✓ Invalid File Handling PASSED (Returned 400 as expected)")

def test_6_database_persistence_and_review():
    print("\n--- Test 6: SQLite Database Persistence & Operator Review ---")
    # Check history
    hist_res = client.get("/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert len(hist) >= 1, "Expected recorded analysis in history"

    # Check detections
    det_res = client.get("/detections")
    assert det_res.status_code == 200
    dets = det_res.json()
    assert len(dets) >= 1, "Expected recorded detections"

    target_det_id = dets[0]["detection_id"]

    # Update review status
    patch_res = client.patch(
        f"/detections/{target_det_id}",
        json={"status": "Confirmed Victim", "notes": "Verified by ground responder"}
    )
    assert patch_res.status_code == 200, f"Review update failed: {patch_res.text}"

    # Verify update persisted
    det_check = client.get("/detections")
    updated = [d for d in det_check.json() if d["detection_id"] == target_det_id][0]
    assert updated["operator_status"] == "Confirmed Victim"
    print(f"✓ Database Persistence PASSED: {target_det_id} marked as Confirmed Victim")

def test_7_emergency_alert_creation():
    print("\n--- Test 7: Emergency Rescue Alert Recording ---")
    response = client.post(
        "/alerts",
        json={
            "detection_id": "TEST-DET-001",
            "analysis_id": "TEST-ANL-001",
            "victim_status": "Confirmed Victim",
            "confidence": 92.5,
            "location_summary": "Floor 2, Corridor East",
            "notes": "Trapped under collapsed dry-wall. High heat signature."
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "alert_id" in data
    assert data["dispatch_status"] == "Prepared (Notification service not configured)"
    print("✓ Emergency Alert Creation PASSED:", data)

def test_8_drone_stream_adapter_status():
    print("\n--- Test 8: Drone Stream Adapter Handling ---")
    status_res = client.get("/drone-status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["connection"]["connected"] is False
    assert status_data["connection"]["status"] == "not_configured"
    print("✓ Drone Stream Adapter PASSED (Honest 'not_configured' status returned)")

def test_9_location_map_integration():
    print("\n--- Test 9: Location Map Markers & Coordinate Update ---")
    # 1. Fetch map markers
    markers_res = client.get("/locations/markers")
    assert markers_res.status_code == 200, f"Map markers fetch failed: {markers_res.text}"
    data = markers_res.json()
    assert "markers" in data
    print(f"✓ Retrieved {data['count']} victim map markers for tactical display")

    # 2. Update victim location
    hist_res = client.get("/history")
    if hist_res.status_code == 200 and len(hist_res.json()) > 0:
        target_id = hist_res.json()[0]["analysis_id"]
        update_res = client.post(
            "/locations/update",
            json={
                "analysis_id": target_id,
                "latitude": 17.3854,
                "longitude": 78.4862,
                "floor": "Floor 2 (Sector B)",
                "room": "Room 204 (Open Workspace)",
                "status": "Camera Lock Pinpoint"
            }
        )
        assert update_res.status_code == 200
        print(f"✓ Victim location updated successfully for {target_id} at (17.3854, 78.4862)")

if __name__ == "__main__":
    test_1_health_endpoint()
    test_2_real_yolo_inference_on_person()
    test_3_real_inference_on_thermal_hand()
    test_4_frame_analysis_for_webcam()
    test_5_invalid_file_handling()
    test_6_database_persistence_and_review()
    test_7_emergency_alert_creation()
    test_8_drone_stream_adapter_status()
    test_9_location_map_integration()
    print("\n=======================================================")
    print("ALL 9 AUTOMATED TESTS PASSED SUCCESSFULLY!")
    print("=======================================================")

