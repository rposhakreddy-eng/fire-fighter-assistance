"""
SQLite Database Layer for Fire Fighter Assistance Drone System
Provides persistent storage for analysis sessions, individual detections,
human victim confirmations, and rescue alerts.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Storage location is configurable so deployments can mount a persistent disk
# (e.g. FIREGUARD_DATA_DIR=/data on Render). Defaults to the backend folder.
DATA_DIR = os.getenv("FIREGUARD_DATA_DIR", os.path.dirname(__file__))
DB_FILE = os.getenv("FIREGUARD_DB_PATH", os.path.join(DATA_DIR, "fireguard.db"))
os.makedirs(os.path.dirname(DB_FILE) or ".", exist_ok=True)


def get_db_connection():
    """Create a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        source_type TEXT NOT NULL,
        filename TEXT NOT NULL,
        original_image_path TEXT NOT NULL,
        annotated_image_path TEXT NOT NULL,
        people_detected INTEGER NOT NULL DEFAULT 0,
        objects_summary TEXT DEFAULT '{}',
        best_confidence REAL NOT NULL DEFAULT 0.0,
        visible_body_parts TEXT DEFAULT '',
        model_used TEXT NOT NULL,
        thermal_palette TEXT DEFAULT 'Normal/Standard',
        gps_latitude REAL,
        gps_longitude REAL,
        floor_number TEXT DEFAULT 'Exact indoor location unavailable',
        room_number TEXT DEFAULT 'Exact indoor location unavailable',
        location_status TEXT DEFAULT 'Location unavailable',
        grok_assessment TEXT DEFAULT '',
        grok_body_part TEXT DEFAULT '',
        grok_evidence TEXT DEFAULT '',
        grok_recommendation TEXT DEFAULT '',
        status TEXT NOT NULL DEFAULT 'completed'
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id TEXT UNIQUE NOT NULL,
        analysis_id TEXT NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL,
        bbox_y1 REAL NOT NULL,
        bbox_x2 REAL NOT NULL,
        bbox_y2 REAL NOT NULL,
        keypoints_json TEXT DEFAULT '[]',
        visible_body_parts TEXT DEFAULT '',
        operator_status TEXT NOT NULL DEFAULT 'Pending Review',
        operator_notes TEXT DEFAULT '',
        timestamp TEXT NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analyses(analysis_id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emergency_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT UNIQUE NOT NULL,
        detection_id TEXT NOT NULL,
        analysis_id TEXT NOT NULL,
        victim_status TEXT NOT NULL,
        confidence REAL NOT NULL,
        location_summary TEXT NOT NULL,
        notes TEXT DEFAULT '',
        dispatch_status TEXT NOT NULL DEFAULT 'Prepared (Service Not Configured)',
        created_at TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()


def save_analysis(
    analysis_id: str,
    source_type: str,
    filename: str,
    original_image_path: str,
    annotated_image_path: str,
    people_detected: int,
    objects_summary: Dict[str, int],
    best_confidence: float,
    visible_body_parts: str,
    model_used: str,
    thermal_palette: str = "Normal/Standard",
    gps_latitude: Optional[float] = None,
    gps_longitude: Optional[float] = None,
    floor_number: str = "Exact indoor location unavailable",
    room_number: str = "Exact indoor location unavailable",
    location_status: str = "Location unavailable",
    grok_assessment: str = "",
    grok_body_part: str = "",
    grok_evidence: str = "",
    grok_recommendation: str = "",
    status: str = "completed"
) -> bool:
    """Save an analysis record to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()

    try:
        cursor.execute("""
        INSERT INTO analyses (
            analysis_id, timestamp, source_type, filename,
            original_image_path, annotated_image_path, people_detected,
            objects_summary, best_confidence, visible_body_parts,
            model_used, thermal_palette, gps_latitude, gps_longitude,
            floor_number, room_number, location_status,
            grok_assessment, grok_body_part, grok_evidence, grok_recommendation, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id, now_iso, source_type, filename,
            original_image_path, annotated_image_path, people_detected,
            json.dumps(objects_summary), best_confidence, visible_body_parts,
            model_used, thermal_palette, gps_latitude, gps_longitude,
            floor_number, room_number, location_status,
            grok_assessment, grok_body_part, grok_evidence, grok_recommendation, status
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Database error saving analysis: {e}")
        return False
    finally:
        conn.close()


def save_detection(
    detection_id: str,
    analysis_id: str,
    class_name: str,
    confidence: float,
    bbox: List[float],
    keypoints: List[Any],
    visible_body_parts: str = "",
    operator_status: str = "Pending Review"
) -> bool:
    """Save an individual object detection record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()

    x1, y1, x2, y2 = bbox if len(bbox) == 4 else (0.0, 0.0, 0.0, 0.0)

    try:
        cursor.execute("""
        INSERT INTO detections (
            detection_id, analysis_id, class_name, confidence,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            keypoints_json, visible_body_parts, operator_status, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection_id, analysis_id, class_name, confidence,
            x1, y1, x2, y2,
            json.dumps(keypoints), visible_body_parts, operator_status, now_iso
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Database error saving detection: {e}")
        return False
    finally:
        conn.close()


def get_all_analyses(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent analysis runs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single analysis and its detections."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    analysis = dict(row)

    cursor.execute("SELECT * FROM detections WHERE analysis_id = ? ORDER BY id ASC", (analysis_id,))
    det_rows = cursor.fetchall()
    analysis["detections"] = [dict(d) for d in det_rows]

    conn.close()
    return analysis


def get_all_detections(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve all detections with analysis metadata."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT d.*, a.filename, a.source_type, a.annotated_image_path, a.original_image_path,
           a.gps_latitude, a.gps_longitude, a.floor_number, a.room_number, a.location_status
    FROM detections d
    LEFT JOIN analyses a ON d.analysis_id = a.analysis_id
    ORDER BY d.id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_detection_status(detection_id: str, status: str, notes: str = "") -> bool:
    """Update human verification status for a detection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        UPDATE detections
        SET operator_status = ?, operator_notes = ?
        WHERE detection_id = ?
        """, (status, notes, detection_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating detection status: {e}")
        return False
    finally:
        conn.close()


def create_emergency_alert(
    alert_id: str,
    detection_id: str,
    analysis_id: str,
    victim_status: str,
    confidence: float,
    location_summary: str,
    notes: str = ""
) -> bool:
    """Record an emergency alert for firefighter dispatch."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.now().isoformat()
    try:
        cursor.execute("""
        INSERT INTO emergency_alerts (
            alert_id, detection_id, analysis_id, victim_status,
            confidence, location_summary, notes, dispatch_status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Prepared (Service Not Configured)', ?)
        """, (alert_id, detection_id, analysis_id, victim_status, confidence, location_summary, notes, now_iso))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error creating emergency alert: {e}")
        return False
    finally:
        conn.close()


def get_all_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve emergency alerts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM emergency_alerts ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_analysis(analysis_id: str) -> bool:
    """Delete an analysis and its associated detections."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM detections WHERE analysis_id = ?", (analysis_id,))
        cursor.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
        cursor.execute("DELETE FROM emergency_alerts WHERE analysis_id = ?", (analysis_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting analysis: {e}")
        return False
    finally:
        conn.close()


def get_system_counts() -> Dict[str, Any]:
    """Get verified stats for the main dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM detections WHERE operator_status = 'Confirmed Victim'")
    confirmed_victims = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE operator_status = 'Possible Victim' OR (class_name = 'person' AND operator_status = 'Pending Review')")
    possible_victims = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analyses")
    total_analyses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]

    cursor.execute("SELECT timestamp FROM analyses ORDER BY id DESC LIMIT 1")
    last_analysis_row = cursor.fetchone()
    last_analysis_time = last_analysis_row[0] if last_analysis_row else None

    conn.close()
    return {
        "confirmed_victims": confirmed_victims,
        "possible_victims": possible_victims,
        "total_analyses": total_analyses,
        "total_detections": total_detections,
        "last_analysis_time": last_analysis_time
    }


def update_analysis_location(
    analysis_id: str,
    lat: Optional[float],
    lon: Optional[float],
    floor: str,
    room: str,
    status: str = "Operator Verified Location"
) -> bool:
    """Update coordinates and indoor floor/room for an analysis session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        UPDATE analyses
        SET gps_latitude = ?, gps_longitude = ?, floor_number = ?, room_number = ?, location_status = ?
        WHERE analysis_id = ?
        """, (lat, lon, floor, room, status, analysis_id))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating analysis location: {e}")
        return False
    finally:
        conn.close()


def get_all_victim_map_markers() -> List[Dict[str, Any]]:
    """Get all detection records with spatial location data for map visualization."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT d.detection_id, d.analysis_id, d.class_name, d.confidence, d.operator_status,
           d.visible_body_parts, d.timestamp,
           a.filename, a.annotated_image_path, a.gps_latitude, a.gps_longitude,
           a.floor_number, a.room_number, a.location_status
    FROM detections d
    JOIN analyses a ON d.analysis_id = a.analysis_id
    WHERE d.class_name = 'person' OR d.operator_status IN ('Possible Victim', 'Confirmed Victim')
    ORDER BY d.id DESC LIMIT 100
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Automatically initialize tables on import
init_db()

