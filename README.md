# 🛸 Fire Fighter Assistance Drone for Victim Detection in Smoke-Filled Buildings
### AI-Powered Search & Rescue Tactical Ground Control Platform
**Author:** Second-Year CSE College Project  
**Platform:** Windows 10/11 | Python 3.12 | FastAPI | PyTorch & Ultralytics YOLO11 | OpenCV | SQLite

---

## 1. Project Overview
In structural fires and smoke-filled building collapses, high-temperature smoke and zero-visibility conditions severely hinder emergency responders. This project delivers an AI-assisted aerial drone software prototype that:
- Ingests thermal imagery, RGB drone video, or live camera feeds.
- Performs genuine local AI computer vision inference using **YOLO11n** (object detection) and **YOLO11n-pose** (17 human skeletal keypoints).
- Detects trapped victims, draws color-coded bounding boxes and anatomical skeletons, and computes true confidence percentages.
- Identifies visible body parts (Head/Face, Torso, Arms/Hands, Legs/Feet) from keypoints.
- Provides secondary multimodal AI reasoning (via Groq Vision) for ambiguous heat signatures and rescue recommendations.
- Persists all analyses, detections, and responder verification audits into a local **SQLite** database (`backend/fireguard.db`).
- Features a dark tactical emergency response dashboard with 10 dedicated functional views inspired by modern search-and-rescue mission interfaces.

---

## 2. Core System Architecture

```
fire fighter assistance drone/
├── backend/
│   ├── main.py              # FastAPI server, REST API endpoints, static file serving
│   ├── detector.py          # YOLO11n detection + pose estimation + OpenCV rendering
│   ├── database.py          # SQLite persistence (analyses, detections, alerts)
│   ├── drone_stream.py      # Webcam frame processor, video sampler, drone stream adapter
│   ├── grok_ai.py           # Multimodal vision AI analysis (Groq fallback)
│   ├── fireguard.db         # Persistent SQLite database
│   └── models/
│       ├── yolo11n.pt       # YOLO11 nano detection model (COCO 80 classes)
│       └── yolo11n-pose.pt  # YOLO11 nano human pose model (17 keypoints)
├── frontend/
│   ├── index.html           # 10-view tactical emergency dashboard
│   ├── style.css            # Dark tactical CSS (glassmorphism, alerts, print stylesheet)
│   └── script.js            # Client logic (real-time polling, webcam, video, DB history)
├── images/                  # Sample test images
│   ├── sample_victim_person.jpg
│   ├── sample_thermal_hand.jpg
│   └── sample_rescue_scene.jpg
├── results/                 # Annotated images, processed video outputs, keyframes
├── tests/
│   └── test_system.py       # Automated 8-point verification test suite
├── requirements.txt         # Python package dependencies
├── .env.example             # Template for configuration
├── .gitignore               # Protects .env secrets and temporary output
└── README.md                # Project documentation and user manual
```

---

## 3. Software Requirements
- **Operating System:** Windows 10 or Windows 11
- **Python:** Python 3.10, 3.11, or 3.12 (Checked on Python 3.12.7)
- **Terminal:** Windows PowerShell
- **IDE:** VS Code or Antigravity IDE
- **Browser:** Google Chrome, Microsoft Edge, or Mozilla Firefox

---

## 4. Installation & Setup Instructions

Open **Windows PowerShell** in your project folder:

```powershell
cd "c:\Users\LENOVO\OneDrive\Desktop\fire fighter assistance drone"
```

### Step 4.1: Verify or Install Python Dependencies
Your system already has PyTorch, Ultralytics YOLO, OpenCV, and FastAPI installed. To ensure all packages are up to date:

```powershell
pip install -r requirements.txt
```

### Step 4.2: Configure Environment Variables
Copy the template to create `.env` (if not already present):

```powershell
Copy-Item .env.example .env
```
*(Your `.env` file already contains your valid `GROQ_API_KEY` for secondary vision reasoning).*

---

## 5. How to Run the Complete System

### Option A: Single-Port Launch (Recommended & Simplest)
Run the backend with FastAPI. It automatically serves both the backend API and the tactical frontend dashboard at `http://127.0.0.1:8000/app/index.html`:

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Now open your web browser and navigate to:
👉 **`http://127.0.0.1:8000/app/index.html`**

---

### Option B: Dual-Terminal Launch (Independent Frontend Server)
If you prefer running the frontend on a separate live-reload server (e.g. Python HTTP server or VS Code Live Server):

**Terminal 1 (Backend API):**
```powershell
cd "c:\Users\LENOVO\OneDrive\Desktop\fire fighter assistance drone"
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend Static Server):**
```powershell
cd "c:\Users\LENOVO\OneDrive\Desktop\fire fighter assistance drone\frontend"
python -m http.server 3000
```
Open your browser at:
👉 **`http://127.0.0.1:3000`**

---

## 6. How to Use the System Features

### 1. Upload & Analyse an Image
1. Click **"Upload & Analyse"** on the left sidebar.
2. Click **"Browse Files"** or click one of the quick test buttons:
   - **"Load Sample Rescue Image"**: Tests multi-person detection + 17 pose keypoints.
   - **"Load Sample Thermal Hand"**: Tests thermal heat signature recognition.
3. Select your desired confidence threshold slider (default: `25%`).
4. Click **"Run AI Victim Analysis"**.
5. View side-by-side original and annotated evidence with labeled bounding boxes and pose skeletons.
6. Click **"Download Annotated Image"** to save evidence.

### 2. Live Laptop Webcam Analysis (Mode A)
1. Click **"Live Drone Feed"** in the sidebar.
2. Select **"Mode A: Laptop Webcam"**.
3. Click **"▶ Start Camera"** and grant browser webcam permission.
4. Raw camera frames are streamed to the local YOLO11 engine at ~2.5 FPS, returning real-time bounding boxes, latency in milliseconds, and detected people counts.
5. Click **"⏹ Stop Camera"** when done.

### 3. Upload & Process a Mission Video (Mode B)
1. Select **"Mode B: Uploaded Video"**.
2. Click **"Browse Video File"** and choose a `.mp4` or `.mov` file.
3. Click **"Process Video Frames"**.
4. The backend samples frames, tracks detections across the timeline, extracts keyframes, and produces an annotated summary video in the `results/` folder.

### 4. Drone RTSP Camera Stream (Mode C)
1. Select **"Mode C: Drone RTSP Stream"**.
2. Enter your drone's RTSP endpoint (e.g. `rtsp://192.168.1.100:554/live/ch0`).
3. Click **"Test Stream Reachability"**.
4. If no drone is connected, the system honestly displays:
   `"Drone stream not configured"` instead of inventing fake video.

### 5. Operator Review & Confirmations (Victim Detections)
1. Navigate to **"Victim Detections"**.
2. Inspect individual detections (`DET-001`, `DET-002`, etc.) with confidence percentages.
3. An operator can explicitly click:
   - **Confirm**: Marks as `Confirmed Victim` (increments Dashboard Total Confirmed Victims).
   - **Possible**: Marks as `Possible Victim`.
   - **Alert**: Opens the **Prepare Rescue Alert** modal.

### 6. Emergency Alerts Dispatch
1. Navigate to **"Rescue Alerts"**.
2. Review prepared alerts containing detection IDs, location summaries, and responder notes.
3. The system honestly reports: `"Notification service not configured"` (external SMS/CAD dispatch credentials remain protected).

### 7. Persistent History
1. Navigate to **"Detection History"**.
2. Search and filter previous drone analysis runs saved in the SQLite database.
3. Click **"View"** to inspect past evidence or **"Delete"** to remove records.

---

## 7. How to Run the Automated Test Suite

We have provided a comprehensive 8-point automated test script to verify all functions:

```powershell
python tests/test_system.py
```

Expected output:
```
✓ Health check PASSED
✓ Person Detection PASSED (2 persons, 89.1% conf, 17 keypoints)
✓ Thermal Analysis PASSED (Hand & forearm identified by Groq)
✓ Frame Analysis PASSED (2.6 FPS, people count)
✓ Invalid File Handling PASSED (Returned HTTP 400)
✓ Database Persistence PASSED (Updated status in SQLite)
✓ Emergency Alert Creation PASSED (Local alert created)
✓ Drone Stream Adapter PASSED (Honest status returned)
=======================================================
ALL 8 AUTOMATED TESTS PASSED SUCCESSFULLY!
=======================================================
```

---

## 8. Features Implemented vs. Known Limitations

### ✅ Fully Implemented Features:
1. **Real YOLO11 Inference:** Person detection with actual bounding boxes, real confidence percentages (never fabricated).
2. **Human Pose Estimation:** 17 COCO skeletal keypoints using `yolo11n-pose.pt`.
3. **Anatomical Body Parts:** Mapped to Head/Face, Torso, Arms/Hands, Legs/Feet based on detected keypoints.
4. **Multimodal Reasoning:** Groq Vision integration for thermal signature interpretation and rescue advice.
5. **Simulated Thermal Colormaps:** OpenCV-based Ironbow/Inferno, Rainbow/Jet, Hot, and White/Black-Hot.
6. **Live Webcam Streaming:** Browser camera capture to backend with real-time annotated frame feedback.
7. **Drone Video Processor:** Video frame extraction, progress indicator, and keyframe snapshots.
8. **Drone Stream Adapter:** Real RTSP/HTTP configuration with connection reachability testing.
9. **SQLite Persistent Database:** Tables for `analyses`, `detections`, and `emergency_alerts`.
10. **Human Operator Verification Workflow:** Distinguishes AI detections from human-confirmed victims.
11. **10-View Emergency Response Dashboard:** Complete dark tactical interface.
12. **Side-by-Side Evidence Viewer & Printable Reports.**

### ⚠️ Known Engineering Limitations (Honest System Disclaimers):
1. **Thermal Validation:** Standard COCO-trained YOLO models detect visual contours; they have not been trained on pure radiometric infrared thermal sensor arrays. Prominently labeled: *"Thermal-image detection has not been validated for this model."*
2. **Temperature Values:** Optical/RGB cameras cannot measure calibrated surface temperatures without a physical FLIR/radiometric infrared sensor. The software displays `"N/A"` for temperatures rather than inventing values.
3. **Indoor Positioning:** Standard drone GPS does not penetrate reinforced concrete structures. Indoor room and floor numbers are labeled: *"Exact indoor location unavailable without indoor positioning/building map integration."*
4. **External Dispatch:** SMS/CAD notification service is labeled `"Not configured"` until an external Twilio or emergency CAD API key is connected.
