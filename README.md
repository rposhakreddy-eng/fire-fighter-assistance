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

---

## 9. Deploying to Render (Cloud)

This repository ships with everything Render needs: a `Dockerfile`, a `render.yaml`
Blueprint, and CPU-only Python dependencies (no multi-gigabyte CUDA wheels).

### 9.1 Before you deploy
Make sure the YOLO model weights are committed to Git — the app cannot detect anyone
without them, and Render builds only from your GitHub repository:

```powershell
git status
# backend/models/yolo11n.pt and backend/models/yolo11n-pose.pt must be listed
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

The two files total ~11 MB. If they do not appear, run
`git add -f backend/models/yolo11n.pt backend/models/yolo11n-pose.pt`.

### 9.2 Deploy with the Blueprint (recommended)
1. Sign in to [render.com](https://render.com) and connect your GitHub account.
2. Click **New +** → **Blueprint**.
3. Select the `fire-fighter-assistance` repository and click **Apply**.
4. When Render asks for `GROQ_API_KEY`, paste your key (optional — without it the
   interface honestly shows *"Not configured"*).
5. Wait for the Docker build to finish (first build takes ~5–10 minutes).
6. Open `https://<your-service-name>.onrender.com` — it redirects automatically to
   the tactical dashboard at `/app/index.html`.

### 9.3 Deploy manually (without Blueprint)
1. **New +** → **Web Service** → choose the repository.
2. Language / Runtime: **Docker**.
3. Health Check Path: `/health`.
4. Add the environment variable `GROQ_API_KEY`.
5. Deploy.

### 9.4 Environment variables
| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Optional | Enables the secondary Groq Vision reasoning. Get one free at [console.groq.com](https://console.groq.com/). |
| `FIREGUARD_DATA_DIR` | Set by Blueprint | Storage root for SQLite + annotated images (defaults to the project folder). |
| `YOLO_IMG_SIZE` | Tuned for free plan | Inference resolution (multiple of 32). `320` on the free plan, `640` for full accuracy. |
| `YOLO_FRAME_IMG_SIZE` | Optional | Live-frame resolution (defaults to 256). Smaller = a frame finishes before Render's proxy timeout. |
| `MALLOC_ARENA_MAX` | Optional | `2` keeps memory usage lower on small instances. |
| `PORT` | Provided by Render | Uvicorn binds to this port automatically. |

### 9.5 Things to know about the free tier
- **CPU is the real limit:** the free plan gives only **0.1 CPU**. A single image
  analysis takes roughly **60–90 seconds**, so this deployment is tuned for it:
  `YOLO_IMG_SIZE=320` and health checks disabled (otherwise Render restarts the
  instance while it is busy inferring).
- **Live webcam mode works at roughly one annotated frame per minute** on the free
  plan (`YOLO_FRAME_IMG_SIZE=256`). The UI shows an elapsed-time counter while it
  waits. Real-time video needs a GPU or a local run — use **Upload & Analyse** for
  a single image (~60–90 s).
- **Render's proxy also drops requests after ~100 seconds.** If an analysis exceeds
  that, lower `YOLO_IMG_SIZE` to `256`.
- **Memory:** 512 MB is enough at 320 px (measured peak ~400 MB). If the service
  restarts, check the logs for an out-of-memory kill.
- **Making it fast:** upgrade to **Standard (1 CPU / 2 GB)**, set
  `YOLO_IMG_SIZE=640`, and optionally re-add `healthCheckPath: /health` in
  `render.yaml`. Inference then takes a few seconds.
- **Cold starts:** free services sleep after ~15 minutes of inactivity; the first
  request afterwards takes ~30–60 seconds while Python and the models load.
- **Ephemeral disk:** SQLite history and annotated images are lost on every deploy
  or restart. To keep them, upgrade the plan, uncomment the `disk:` block in
  `render.yaml` (mount path `/data`), and set `FIREGUARD_DATA_DIR=/data`.
- **HTTPS:** Render provides it, which is required by the browser for webcam access.
- **Region:** change `region: oregon` in `render.yaml` to the closest region
  (e.g. `singapore`) before deploying.

### 9.6 Local development is unchanged
```powershell
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
On Linux/macOS the same command works; the dependencies are now headless and
CPU-only, so they behave identically on a server.
