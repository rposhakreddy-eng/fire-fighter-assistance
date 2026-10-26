/**
 * Fire Fighter Assistance Drone — Main Client Application Script
 * Connects directly to local FastAPI backend (http://127.0.0.1:8000)
 * Handles real YOLO11 inference, webcam capture, video analysis, SQLite history, and operator reviews.
 */

function getInitialApiBase() {
    const saved = localStorage.getItem("FIRE_FIGHTER_API_BASE");
    if (saved) return saved;
    // When served directly by the backend (localhost, local Wi-Fi IP, or public tunnel like ngrok/localtunnel)
    if (window.location.origin && window.location.origin.startsWith("http") && !window.location.hostname.includes("netlify.app")) {
        return window.location.origin;
    }
    return "http://127.0.0.1:8000";
}
let API_BASE = getInitialApiBase();

// State Management
const AppState = {
    activeView: "dashboard",
    currentAnalysis: null,
    selectedFile: null,
    backendOnline: false,
    webcamStream: null,
    webcamInterval: null,
    isWebcamRunning: false,
    confidenceThreshold: 0.25,
    activeModel: "yolo11n-pose.pt",
    droneStreamConfig: { url: "", protocol: "RTSP" },
    deleteTargetId: null,
    lastFrameBase64: null
};

// ==============================================================
// INITIALIZATION
// ==============================================================
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initNavigation();
    initUploadHandlers();
    initWebcamHandlers();
    initVideoHandlers();
    initDroneStreamHandlers();
    initThermalPaletteHandlers();
    initDetectionsHandlers();
    initAlertsHandlers();
    initHistoryHandlers();
    initSettingsHandlers();
    initSampleDataButtons();
    TacticalMap.init();

    // Start background health polling
    checkBackendHealth();
    setInterval(checkBackendHealth, 6000);
});

// ==============================================================
// CLOCK & HEALTH CHECK
// ==============================================================
function initClock() {
    const clockEl = document.getElementById("systemClock");
    function update() {
        const now = new Date();
        clockEl.textContent = now.toUTCString().split(" ")[4] + " UTC";
    }
    update();
    setInterval(update, 1000);
}

async function checkBackendHealth() {
    const backendStatusText = document.getElementById("backendStatusText");
    const backendDot = document.getElementById("backendDot");
    const cardDroneStatus = document.getElementById("cardDroneStatus");
    const cardDroneFootnote = document.getElementById("cardDroneFootnote");

    try {
        const res = await fetch(`${API_BASE}/health`, { method: "GET" });
        if (res.ok) {
            const data = await res.json();
            AppState.backendOnline = true;
            backendStatusText.textContent = "ONLINE";
            backendDot.className = "status-dot dot-green dot-pulse";

            // Update stats if available
            if (data.stats) {
                document.getElementById("cardConfirmedVictims").textContent = data.stats.confirmed_victims || 0;
                document.getElementById("cardPossibleVictims").textContent = data.stats.possible_victims || 0;
                if (data.stats.last_analysis_time) {
                    const d = new Date(data.stats.last_analysis_time);
                    document.getElementById("cardLastTime").textContent = d.toLocaleTimeString();
                }
            }

            // Update drone status pill
            const droneText = document.getElementById("droneStreamText");
            const droneDot = document.getElementById("droneDot");
            if (data.drone_stream && data.drone_stream.connected) {
                droneText.textContent = "CONNECTED";
                droneDot.className = "status-dot dot-green";
                cardDroneStatus.textContent = "Connected";
                cardDroneFootnote.textContent = "Live telemetry active";
            } else {
                droneText.textContent = "STANDBY";
                droneDot.className = "status-dot dot-amber";
                cardDroneStatus.textContent = "Standby";
                cardDroneFootnote.textContent = "Drone stream not configured";
            }

            // Update Settings Groq indicator
            const settingGroqStatus = document.getElementById("settingGroqStatus");
            if (settingGroqStatus && data.groq_ai) {
                settingGroqStatus.textContent = data.groq_ai.status;
                settingGroqStatus.className = data.groq_ai.configured ? "text-green" : "text-amber";
            }

            const settingApiHealth = document.getElementById("settingApiHealthText");
            if (settingApiHealth) {
                settingApiHealth.textContent = `ONLINE — Connected to ${API_BASE}`;
                settingApiHealth.className = "text-green";
            }
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch (err) {
        AppState.backendOnline = false;
        backendStatusText.textContent = "OFFLINE";
        backendDot.className = "status-dot dot-red";
        cardDroneStatus.textContent = "Disconnected";
        cardDroneFootnote.textContent = "Backend offline";

        const settingApiHealth = document.getElementById("settingApiHealthText");
        if (settingApiHealth) {
            settingApiHealth.textContent = `OFFLINE (${API_BASE}) — Start FastAPI backend on port 8000 or enter live URL`;
            settingApiHealth.className = "text-red";
        }
    }
}

// ==============================================================
// NAVIGATION & VIEW SWITCHING
// ==============================================================
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetView = item.getAttribute("data-view");
            switchView(targetView);
        });
    });

    // Dashboard quick action buttons
    document.getElementById("backendStatusPill")?.addEventListener("click", () => switchView("settings"));
    document.getElementById("dashUploadBtn")?.addEventListener("click", () => switchView("upload"));
    document.getElementById("dashLiveFeedBtn")?.addEventListener("click", () => switchView("livefeed"));
    document.getElementById("dashEmptyUploadBtn")?.addEventListener("click", () => switchView("upload"));
    document.getElementById("dashEmptyWebcamBtn")?.addEventListener("click", () => {
        switchView("livefeed");
        document.getElementById("startWebcamBtn")?.click();
    });
    document.getElementById("dashViewEvidenceBtn")?.addEventListener("click", () => switchView("evidence"));
}

function switchView(viewId) {
    AppState.activeView = viewId;

    // Update active nav button
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-view") === viewId);
    });

    // Update active view panel
    document.querySelectorAll(".view-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `view-${viewId}`);
    });

    // Trigger on-enter view refreshes
    if (viewId === "detections") loadDetectionsTable();
    if (viewId === "history") loadHistoryTable();
    if (viewId === "alerts") loadAlertsTable();
    if (viewId === "location") {
        setTimeout(() => {
            if (tacticalMap) tacticalMap.invalidateSize();
        }, 120);
    }
}

// ==============================================================
// IMAGE UPLOAD & AI INFERENCE PIPELINE
// ==============================================================
function initUploadHandlers() {
    const mainImageInput = document.getElementById("mainImageInput");
    const mainUploadBtn = document.getElementById("mainUploadBtn");
    const mainDropZone = document.getElementById("mainDropZone");
    const runAnalysisBtn = document.getElementById("runAnalysisBtn");
    const clearAnalysisBtn = document.getElementById("clearAnalysisBtn");
    const confSlider = document.getElementById("uploadConfidenceSlider");
    const confLabel = document.getElementById("confValueLabel");

    confSlider?.addEventListener("input", (e) => {
        confLabel.textContent = `${e.target.value}%`;
        AppState.confidenceThreshold = parseFloat(e.target.value) / 100.0;
    });

    mainUploadBtn?.addEventListener("click", () => mainImageInput.click());

    mainImageInput?.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedImage(e.target.files[0]);
        }
    });

    // Drag and drop
    mainDropZone?.addEventListener("dragover", (e) => {
        e.preventDefault();
        mainDropZone.style.borderColor = "var(--accent-blue)";
    });
    mainDropZone?.addEventListener("dragleave", () => {
        mainDropZone.style.borderColor = "var(--border-subtle)";
    });
    mainDropZone?.addEventListener("drop", (e) => {
        e.preventDefault();
        mainDropZone.style.borderColor = "var(--border-subtle)";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedImage(e.dataTransfer.files[0]);
        }
    });

    runAnalysisBtn?.addEventListener("click", runImageInference);
    clearAnalysisBtn?.addEventListener("click", clearUploadState);
}

function handleSelectedImage(file) {
    if (!file.type.startsWith("image/")) {
        showToast("Please select a valid image file (JPEG, PNG, WebP).", "error");
        return;
    }

    AppState.selectedFile = file;
    document.getElementById("mainSelectedFileName").textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

    // Preview original image
    const origPreview = document.getElementById("uploadOrigPreview");
    const origPlaceholder = document.getElementById("origPlaceholder");
    const objectUrl = URL.createObjectURL(file);
    origPreview.src = objectUrl;
    origPreview.style.display = "block";
    origPlaceholder.style.display = "none";

    document.getElementById("runAnalysisBtn").disabled = false;
    document.getElementById("uploadStatusBadge").textContent = "Ready to Analyse";
    document.getElementById("uploadStatusBadge").className = "badge badge-amber";
}

async function runImageInference() {
    if (!AppState.selectedFile) {
        showToast("Please upload an image first.", "warning");
        return;
    }

    const runBtn = document.getElementById("runAnalysisBtn");
    runBtn.disabled = true;
    runBtn.textContent = "Running YOLO11 Inference...";
    document.getElementById("uploadStatusBadge").textContent = "Analysing...";

    const formData = new FormData();
    formData.append("file", AppState.selectedFile);
    formData.append("confidence_threshold", AppState.confidenceThreshold);
    formData.append("palette", document.getElementById("uploadPaletteSelect").value);

    const lat = document.getElementById("uploadGpsLat").value.trim();
    const lon = document.getElementById("uploadGpsLon").value.trim();
    if (lat) formData.append("gps_lat", lat);
    if (lon) formData.append("gps_lon", lon);

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        AppState.currentAnalysis = data;

        // Render Results across Dashboard, Upload, Evidence, Location views
        renderAnalysisResults(data);
        showToast(`Inference complete! ${data.people_detected} person(s) detected.`, "success");

    } catch (err) {
        console.error("Inference error:", err);
        showToast(`Inference failed: ${err.message}`, "error");
        document.getElementById("uploadStatusBadge").textContent = "Inference Failed";
        document.getElementById("uploadStatusBadge").className = "badge badge-red";
    } finally {
        runBtn.disabled = false;
        runBtn.textContent = "Run AI Victim Analysis";
    }
}

function renderAnalysisResults(data) {
    // 1. Upload View Updates
    const annotatedPreview = document.getElementById("uploadAnnotatedPreview");
    const annotatedPlaceholder = document.getElementById("annotatedPlaceholder2");
    annotatedPreview.src = `${API_BASE}${data.annotated_image_url}?t=${Date.now()}`;
    annotatedPreview.style.display = "block";
    annotatedPlaceholder.style.display = "none";

    document.getElementById("uploadMetricsSummary").style.display = "flex";
    document.getElementById("upPeopleCount").textContent = data.people_detected;
    document.getElementById("upBestConf").textContent = `${data.best_confidence}%`;
    document.getElementById("upLatency").textContent = `${data.inference_time_ms} ms`;
    document.getElementById("uploadDownloadBar").style.display = "flex";
    document.getElementById("uploadStatusBadge").textContent = "Completed";
    document.getElementById("uploadStatusBadge").className = "badge badge-green";

    // 2. Dashboard View Updates
    const dashDisplayImage = document.getElementById("dashDisplayImage");
    const dashEmptyState = document.getElementById("dashEmptyState");
    dashDisplayImage.src = `${API_BASE}${data.annotated_image_url}?t=${Date.now()}`;
    dashDisplayImage.style.display = "block";
    dashEmptyState.style.display = "none";

    document.getElementById("dashFrameBadge").textContent = `${data.people_detected} Person(s) Detected`;
    document.getElementById("dashFrameBadge").className = data.people_detected > 0 ? "badge badge-red" : "badge badge-green";

    // Metrics Cards
    document.getElementById("cardVisiblePeople").textContent = data.people_detected;
    document.getElementById("cardVisibleFootnote").textContent = `In frame: ${data.filename}`;
    document.getElementById("cardConfidence").textContent = data.best_confidence > 0 ? `${data.best_confidence}%` : "0.0%";
    document.getElementById("cardConfidenceFootnote").textContent = `Latency: ${data.inference_time_ms} ms`;
    document.getElementById("cardLastTime").textContent = new Date().toLocaleTimeString();

    // Body parts list
    const bodyPartsContainer = document.getElementById("dashBodyPartsList");
    bodyPartsContainer.innerHTML = "";
    if (data.visible_body_parts && data.visible_body_parts.length > 0) {
        data.visible_body_parts.forEach(part => {
            const chip = document.createElement("span");
            chip.className = "tag-chip";
            chip.textContent = `✓ ${part}`;
            bodyPartsContainer.appendChild(chip);
        });
    } else {
        bodyPartsContainer.innerHTML = '<span class="tag-chip tag-muted">No anatomical regions identified</span>';
    }

    // Groq Multimodal Reasoning
    const groq = data.secondary_ai_analysis;
    if (groq) {
        document.getElementById("dashGroqAssessment").textContent = groq.human_assessment || "N/A";
        document.getElementById("dashGroqPart").textContent = groq.visible_body_part || "N/A";
        document.getElementById("dashGroqEvidence").textContent = groq.evidence || "No evidence recorded.";
        document.getElementById("dashGroqRec").textContent = groq.recommendation || "Follow standard procedure.";
    }

    // Location Fields
    if (data.location) {
        const latLonStr = (data.location.latitude && data.location.longitude)
            ? `${data.location.latitude}, ${data.location.longitude}`
            : "Unavailable";
        document.getElementById("dashGpsCoords").textContent = latLonStr;
        document.getElementById("dashFloor").textContent = data.location.floor;
        document.getElementById("dashRoom").textContent = data.location.room;

        document.getElementById("locLatDisplay").textContent = data.location.latitude || "Unavailable";
        document.getElementById("locLonDisplay").textContent = data.location.longitude || "Unavailable";
        document.getElementById("locFloorDisplay").textContent = data.location.floor;
        document.getElementById("locRoomDisplay").textContent = data.location.room;
        document.getElementById("locSourceDisplay").textContent = data.source_type || "Drone Image Upload";
        document.getElementById("locStatusDisplay").textContent = data.location.status;

        if (data.location.latitude && data.location.longitude) {
            TacticalMap.setVictimCoordinates(parseFloat(data.location.latitude), parseFloat(data.location.longitude));
        }
        if (data.location.floor && data.location.floor !== "Exact indoor location unavailable") {
            TacticalMap.activeFloor = data.location.floor;
        }
        if (data.location.room && data.location.room !== "Exact indoor location unavailable") {
            TacticalMap.activeRoom = data.location.room;
        }
    }

    // Enable Evidence View
    const evidenceOrigImg = document.getElementById("evidenceOrigImg");
    const evidenceAnnotatedImg = document.getElementById("evidenceAnnotatedImg");
    evidenceOrigImg.src = `${API_BASE}${data.original_image_url}`;
    evidenceOrigImg.style.display = "block";
    document.getElementById("evidenceOrigPlaceholder").style.display = "none";
    document.getElementById("downloadOrigBtn").disabled = false;

    evidenceAnnotatedImg.src = `${API_BASE}${data.annotated_image_url}?t=${Date.now()}`;
    evidenceAnnotatedImg.style.display = "block";
    document.getElementById("evidenceAnnotatedPlaceholder").style.display = "none";
    document.getElementById("downloadAnnotatedBtn2").disabled = false;

    document.getElementById("repAnalysisId").textContent = data.analysis_id;
    document.getElementById("repModel").textContent = data.model_used;
    document.getElementById("repPeople").textContent = data.people_detected;
    document.getElementById("repConfidence").textContent = `${data.best_confidence}%`;
    document.getElementById("repBodyParts").textContent = (data.visible_body_parts || []).join(", ") || "None";
    document.getElementById("repLocation").textContent = data.location.status;
    document.getElementById("repNotes").textContent = groq ? `${groq.evidence} ${groq.recommendation}` : "No secondary assessment.";
    document.getElementById("reportTimestamp").textContent = new Date().toLocaleString();

    document.getElementById("dashViewEvidenceBtn").disabled = false;
    document.getElementById("dashExportReportBtn").disabled = false;

    // Refresh persistent numbers
    checkBackendHealth();
}

function clearUploadState() {
    AppState.selectedFile = null;
    document.getElementById("mainImageInput").value = "";
    document.getElementById("mainSelectedFileName").textContent = "";
    document.getElementById("uploadOrigPreview").style.display = "none";
    document.getElementById("origPlaceholder").style.display = "block";
    document.getElementById("uploadAnnotatedPreview").style.display = "none";
    document.getElementById("annotatedPlaceholder2").style.display = "block";
    document.getElementById("uploadMetricsSummary").style.display = "none";
    document.getElementById("uploadDownloadBar").style.display = "none";
    document.getElementById("runAnalysisBtn").disabled = true;
    document.getElementById("uploadStatusBadge").textContent = "Awaiting Image";
    document.getElementById("uploadStatusBadge").className = "badge";
}

// ==============================================================
// MODE A: WEBCAM LIVE STREAMING
// ==============================================================
function initWebcamHandlers() {
    const startBtn = document.getElementById("startWebcamBtn");
    const stopBtn = document.getElementById("stopWebcamBtn");
    const videoEl = document.getElementById("webcamVideo");
    const canvasEl = document.getElementById("webcamCanvas");
    const annotatedImg = document.getElementById("webcamAnnotatedImg");
    const rawPlaceholder = document.getElementById("webcamPlaceholder");
    const annotatedPlaceholder = document.getElementById("annotatedPlaceholder");
    const rawCamBadge = document.getElementById("rawCamBadge");
    const statusText = document.getElementById("webcamStatus");
    const fpsText = document.getElementById("webcamFps");
    const latencyText = document.getElementById("webcamLatency");
    const peopleText = document.getElementById("webcamPeopleCount");

    startBtn?.addEventListener("click", async () => {
        try {
            statusText.textContent = "Requesting camera access...";
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false
            });

            AppState.webcamStream = stream;
            AppState.isWebcamRunning = true;
            videoEl.srcObject = stream;
            videoEl.style.display = "block";
            rawPlaceholder.style.display = "none";
            rawCamBadge.textContent = "ONLINE";
            rawCamBadge.className = "badge badge-green";

            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusText.textContent = "Streaming to YOLO11 Engine...";

            // Setup canvas
            canvasEl.width = 640;
            canvasEl.height = 480;
            const ctx = canvasEl.getContext("2d");

            let isProcessingFrame = false;

            // Stream frames at ~2.5 FPS (every 400ms) to ensure smooth laptop execution
            AppState.webcamInterval = setInterval(async () => {
                if (!AppState.isWebcamRunning || isProcessingFrame) return;

                try {
                    isProcessingFrame = true;
                    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
                    const b64Data = canvasEl.toDataURL("image/jpeg", 0.75);

                    const t0 = performance.now();
                    const res = await fetch(`${API_BASE}/analyze-frame`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            image_base64: b64Data,
                            confidence_threshold: AppState.confidenceThreshold
                        })
                    });

                    if (res.ok) {
                        const frameData = await res.json();
                        const latency = Math.round(performance.now() - t0);

                        if (frameData.annotated_frame_base64) {
                            annotatedImg.src = frameData.annotated_frame_base64;
                            annotatedImg.style.display = "block";
                            annotatedPlaceholder.style.display = "none";
                        }

                        fpsText.textContent = frameData.fps || "2.5";
                        latencyText.textContent = `${latency} ms`;
                        peopleText.textContent = frameData.people_count;

                        // Also update dashboard cards if on dashboard
                        document.getElementById("cardVisiblePeople").textContent = frameData.people_count;
                        document.getElementById("cardConfidence").textContent = frameData.best_confidence > 0 ? `${frameData.best_confidence}%` : "0.0%";
                    }
                } catch (frameErr) {
                    console.warn("Frame inference glitch:", frameErr);
                } finally {
                    isProcessingFrame = false;
                }
            }, 400);

            showToast("Laptop webcam connected to AI detection engine.", "success");

        } catch (camErr) {
            console.error("Camera access error:", camErr);
            statusText.textContent = "Camera access denied or unavailable";
            showToast(`Camera error: ${camErr.message}. Ensure browser permissions are granted.`, "error");
        }
    });

    stopBtn?.addEventListener("click", stopWebcamFeed);
}

function stopWebcamFeed() {
    if (AppState.webcamStream) {
        AppState.webcamStream.getTracks().forEach(track => track.stop());
        AppState.webcamStream = null;
    }
    if (AppState.webcamInterval) {
        clearInterval(AppState.webcamInterval);
        AppState.webcamInterval = null;
    }
    AppState.isWebcamRunning = false;

    document.getElementById("webcamVideo").style.display = "none";
    document.getElementById("webcamPlaceholder").style.display = "block";
    document.getElementById("rawCamBadge").textContent = "OFFLINE";
    document.getElementById("rawCamBadge").className = "badge";

    document.getElementById("startWebcamBtn").disabled = false;
    document.getElementById("stopWebcamBtn").disabled = true;
    document.getElementById("webcamStatus").textContent = "Camera Stopped";
    document.getElementById("webcamFps").textContent = "0.0";
}

// ==============================================================
// MODE B: VIDEO FILE PROCESSING
// ==============================================================
function initVideoHandlers() {
    const videoFileInput = document.getElementById("videoFileInput");
    const selectVideoBtn = document.getElementById("selectVideoBtn");
    const videoDropZone = document.getElementById("videoDropZone");
    const processVideoBtn = document.getElementById("processVideoBtn");
    const clearVideoBtn = document.getElementById("clearVideoBtn");
    const selectedVideoName = document.getElementById("selectedVideoName");

    let videoFile = null;

    selectVideoBtn?.addEventListener("click", () => videoFileInput.click());

    videoFileInput?.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            videoFile = e.target.files[0];
            selectedVideoName.textContent = `Selected: ${videoFile.name} (${(videoFile.size / (1024 * 1024)).toFixed(1)} MB)`;
            processVideoBtn.disabled = false;
        }
    });

    clearVideoBtn?.addEventListener("click", () => {
        videoFile = null;
        videoFileInput.value = "";
        selectedVideoName.textContent = "";
        processVideoBtn.disabled = true;
        document.getElementById("videoProgressCard").style.display = "none";
        document.getElementById("videoResultsArea").style.display = "none";
    });

    processVideoBtn?.addEventListener("click", async () => {
        if (!videoFile) return;

        processVideoBtn.disabled = true;
        const progressCard = document.getElementById("videoProgressCard");
        const progressBar = document.getElementById("videoProgressBar");
        const progressText = document.getElementById("videoProgressText");

        progressCard.style.display = "block";
        progressBar.style.width = "40%";
        progressText.textContent = "Uploading video & running frame-by-frame YOLO11 tracking...";

        const formData = new FormData();
        formData.append("file", videoFile);
        formData.append("confidence_threshold", AppState.confidenceThreshold);
        formData.append("sample_interval", 5);

        try {
            const res = await fetch(`${API_BASE}/analyze-video`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Video processing failed.");
            }

            const data = await res.json();
            progressBar.style.width = "100%";
            progressText.textContent = "Processing completed successfully!";

            document.getElementById("videoResultsArea").style.display = "block";
            document.getElementById("vidTotalFrames").textContent = data.total_video_frames;
            document.getElementById("vidAnalyzedFrames").textContent = data.analyzed_frames;
            document.getElementById("vidMaxPeople").textContent = data.max_people_in_single_frame;
            document.getElementById("vidEventCount").textContent = data.detection_events_count;

            showToast("Drone video frames analyzed successfully!", "success");

        } catch (err) {
            console.error("Video processing error:", err);
            progressBar.style.width = "0%";
            progressText.textContent = `Error: ${err.message}`;
            showToast(`Video error: ${err.message}`, "error");
        } finally {
            processVideoBtn.disabled = false;
        }
    });

    // Mode Selector Tabs (Mode A, Mode B, Mode C)
    const modeBtns = document.querySelectorAll(".btn-mode");
    modeBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            modeBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const mode = btn.getAttribute("data-mode");

            document.getElementById("feedModeWebcam").style.display = mode === "webcam" ? "block" : "none";
            document.getElementById("feedModeVideo").style.display = mode === "video" ? "block" : "none";
            document.getElementById("feedModeDrone").style.display = mode === "drone" ? "block" : "none";

            // If switching away from webcam, stop camera to release hardware
            if (mode !== "webcam" && AppState.isWebcamRunning) {
                stopWebcamFeed();
            }
        });
    });
}

// ==============================================================
// MODE C: REAL DRONE STREAM ADAPTER
// ==============================================================
function initDroneStreamHandlers() {
    const connectBtn = document.getElementById("connectDroneStreamBtn");
    const testBtn = document.getElementById("testDroneStreamBtn");
    const urlInput = document.getElementById("droneStreamUrlInput");
    const protocolSelect = document.getElementById("droneStreamProtocolSelect");
    const statusMsg = document.getElementById("droneStreamStatusMsg");

    async function testStream() {
        const url = urlInput.value.trim();
        if (!url) {
            statusMsg.textContent = "Drone stream not configured";
            statusMsg.className = "status-msg text-amber";
            showToast("Enter a drone stream URL first.", "warning");
            return;
        }

        statusMsg.textContent = "Testing stream reachability...";
        statusMsg.className = "status-msg text-amber";

        try {
            const res = await fetch(`${API_BASE}/drone-config`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url, protocol: protocolSelect.value })
            });

            const data = await res.json();
            if (data.connection && data.connection.connected) {
                statusMsg.textContent = "Drone stream connected!";
                statusMsg.className = "status-msg text-green";
                showToast("Stream reachable! Drone camera connected.", "success");
            } else {
                statusMsg.textContent = data.connection.message || "Unable to reach stream";
                statusMsg.className = "status-msg text-amber";
                showToast(data.connection.message, "warning");
            }
        } catch (err) {
            statusMsg.textContent = `Connection error: ${err.message}`;
            statusMsg.className = "status-msg text-red";
        }
    }

    connectBtn?.addEventListener("click", testStream);
    testBtn?.addEventListener("click", testStream);
}

// ==============================================================
// THERMAL PALETTE SIMULATION
// ==============================================================
function initThermalPaletteHandlers() {
    const paletteButtons = document.querySelectorAll(".palette-buttons-row .btn");
    const paletteImg = document.getElementById("thermalPaletteImg");
    const placeholder = document.getElementById("thermalPalettePlaceholder");

    paletteButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            paletteButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const palette = btn.getAttribute("data-palette");

            if (AppState.currentAnalysis && AppState.currentAnalysis.original_image_url) {
                // If we already have an image analyzed, re-run with this palette
                document.getElementById("uploadPaletteSelect").value = palette;
                showToast(`Switched thermal simulator to: ${palette.toUpperCase()}`, "info");
            } else {
                showToast("Upload an image in 'Upload & Analyse' to view simulated palettes.", "info");
            }
        });
    });
}

// ==============================================================
// VICTIM DETECTIONS TABLE & AUDIT
// ==============================================================
function initDetectionsHandlers() {
    const refreshBtn = document.getElementById("refreshDetectionsBtn");
    const filterSelect = document.getElementById("detectionStatusFilter");

    refreshBtn?.addEventListener("click", loadDetectionsTable);
    filterSelect?.addEventListener("change", loadDetectionsTable);
}

async function loadDetectionsTable() {
    const tbody = document.getElementById("detectionsTableBody");
    const filter = document.getElementById("detectionStatusFilter")?.value || "ALL";

    try {
        const res = await fetch(`${API_BASE}/detections`);
        if (!res.ok) throw new Error("Failed to load detections");
        let detections = await res.json();

        // Update badge count
        document.getElementById("detectionCountBadge").textContent = detections.length;

        if (filter !== "ALL") {
            detections = detections.filter(d => d.operator_status === filter);
        }

        if (detections.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No detections matching the current filter.</td></tr>';
            return;
        }

        tbody.innerHTML = "";
        detections.forEach(det => {
            const tr = document.createElement("tr");

            // Status styling
            let statusBadge = `<span class="badge">${det.operator_status}</span>`;
            if (det.operator_status === "Confirmed Victim") statusBadge = `<span class="badge badge-green">Confirmed Victim</span>`;
            else if (det.operator_status === "Possible Victim") statusBadge = `<span class="badge badge-amber">Possible Victim</span>`;
            else if (det.operator_status === "False Positive") statusBadge = `<span class="badge badge-red">False Positive</span>`;

            const bboxStr = `[${Math.round(det.bbox_x1)}, ${Math.round(det.bbox_y1)}, ${Math.round(det.bbox_x2)}, ${Math.round(det.bbox_y2)}]`;
            const confPct = `${(det.confidence * 100).toFixed(1)}%`;
            const timeStr = new Date(det.timestamp).toLocaleTimeString();

            tr.innerHTML = `
                <td><strong>${det.detection_id}</strong></td>
                <td>${det.class_name}</td>
                <td><span class="text-green">${confPct}</span></td>
                <td><code>${bboxStr}</code></td>
                <td>${det.visible_body_parts || "None"}</td>
                <td>${timeStr}</td>
                <td>${statusBadge}</td>
                <td>
                    <div style="display: flex; gap: 4px;">
                        <button class="btn btn-sm btn-secondary" onclick="updateDetectionReview('${det.detection_id}', 'Confirmed Victim')">Confirm</button>
                        <button class="btn btn-sm btn-secondary" onclick="updateDetectionReview('${det.detection_id}', 'Possible Victim')">Possible</button>
                        <button class="btn btn-sm btn-outline" onclick="locateDetectionOnMap('${det.detection_id}', ${det.gps_latitude || 17.3850}, ${det.gps_longitude || 78.4867}, '${det.floor_number || 'Floor 2'}', '${det.room_number || 'Room 204'}')">📍 Map</button>
                        <button class="btn btn-sm btn-outline" onclick="openAlertModal('${det.detection_id}', '${det.analysis_id}', '${det.confidence}')">Alert</button>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center text-red">Error loading detections: ${err.message}</td></tr>`;
    }
}

window.locateDetectionOnMap = function(detId, lat, lon, floor, room) {
    switchView("location");
    TacticalMap.setVictimCoordinates(lat, lon);
    TacticalMap.activeFloor = floor;
    TacticalMap.activeRoom = room;
    document.getElementById("locFloorDisplay").textContent = floor;
    document.getElementById("locRoomDisplay").textContent = room;
    document.getElementById("dashFloor").textContent = floor;
    document.getElementById("dashRoom").textContent = room;
    TacticalMap.centerOnVictim();
    showToast(`Pinpointed detection ${detId} on tactical map!`, "info");
};

window.updateDetectionReview = async function (detId, status) {
    try {
        const res = await fetch(`${API_BASE}/detections/${detId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status: status, notes: "Operator verified in tactical view." })
        });
        if (res.ok) {
            showToast(`Detection ${detId} marked as ${status}.`, "success");
            loadDetectionsTable();
            checkBackendHealth();
        }
    } catch (err) {
        showToast(`Failed to update status: ${err.message}`, "error");
    }
};

// ==============================================================
// EMERGENCY ALERTS
// ==============================================================
function initAlertsHandlers() {
    const manualBtn = document.getElementById("manualCreateAlertBtn");
    const closeBtn = document.getElementById("closeAlertModalBtn");
    const cancelBtn = document.getElementById("cancelAlertModalBtn");
    const submitBtn = document.getElementById("submitAlertBtn");
    const sendToAlertBtn = document.getElementById("sendToAlertBtn");

    manualBtn?.addEventListener("click", () => {
        openAlertModal(
            AppState.currentAnalysis ? AppState.currentAnalysis.analysis_id : "MANUAL-001",
            AppState.currentAnalysis ? AppState.currentAnalysis.analysis_id : "MANUAL-001",
            AppState.currentAnalysis ? AppState.currentAnalysis.best_confidence : 90.0
        );
    });

    sendToAlertBtn?.addEventListener("click", () => {
        if (!AppState.currentAnalysis) return;
        openAlertModal(
            AppState.currentAnalysis.analysis_id,
            AppState.currentAnalysis.analysis_id,
            AppState.currentAnalysis.best_confidence
        );
    });

    closeBtn?.addEventListener("click", closeAlertModal);
    cancelBtn?.addEventListener("click", closeAlertModal);

    submitBtn?.addEventListener("click", async () => {
        const detId = document.getElementById("alertModalDetId").value;
        const status = document.getElementById("alertModalStatus").value;
        const loc = document.getElementById("alertModalLocation").value.trim() || "Exact indoor location unavailable";
        const notes = document.getElementById("alertModalNotes").value.trim();

        try {
            const res = await fetch(`${API_BASE}/alerts`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    detection_id: detId,
                    analysis_id: AppState.currentAnalysis ? AppState.currentAnalysis.analysis_id : detId,
                    victim_status: status,
                    confidence: AppState.currentAnalysis ? AppState.currentAnalysis.best_confidence : 85.0,
                    location_summary: loc,
                    notes: notes
                })
            });

            if (res.ok) {
                const data = await res.json();
                closeAlertModal();
                showToast(`Alert recorded: ${data.alert_id} (Dispatch service not configured)`, "info");
                loadAlertsTable();
                switchView("alerts");
            }
        } catch (err) {
            showToast(`Alert creation failed: ${err.message}`, "error");
        }
    });
}

window.openAlertModal = function (detId, analysisId, conf) {
    document.getElementById("alertModalDetId").value = detId;
    document.getElementById("alertModalLocation").value = AppState.currentAnalysis && AppState.currentAnalysis.location
        ? `${AppState.currentAnalysis.location.floor}, ${AppState.currentAnalysis.location.room}`
        : "Exact indoor location unavailable";
    document.getElementById("alertModal").style.display = "flex";
};

function closeAlertModal() {
    document.getElementById("alertModal").style.display = "none";
}

async function loadAlertsTable() {
    const tbody = document.getElementById("alertsTableBody");
    try {
        const res = await fetch(`${API_BASE}/alerts`);
        if (!res.ok) throw new Error("Failed to load alerts");
        const alerts = await res.json();

        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No emergency alerts recorded yet.</td></tr>';
            return;
        }

        tbody.innerHTML = "";
        alerts.forEach(alt => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${alt.alert_id}</strong></td>
                <td>${alt.detection_id}</td>
                <td><span class="badge badge-amber">${alt.victim_status}</span></td>
                <td>${alt.confidence}%</td>
                <td>${alt.location_summary}</td>
                <td><span class="badge">${alt.dispatch_status}</span></td>
                <td>${new Date(alt.created_at).toLocaleTimeString()}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-red">Error: ${err.message}</td></tr>`;
    }
}

// ==============================================================
// DETECTION HISTORY
// ==============================================================
function initHistoryHandlers() {
    const searchInput = document.getElementById("historySearchInput");
    const refreshBtn = document.getElementById("historyRefreshBtn");
    const closeDelModalBtn = document.getElementById("closeDeleteModalBtn");
    const cancelDelModalBtn = document.getElementById("cancelDeleteModalBtn");
    const confirmDelBtn = document.getElementById("confirmDeleteBtn");

    refreshBtn?.addEventListener("click", loadHistoryTable);
    searchInput?.addEventListener("input", filterHistoryTable);

    closeDelModalBtn?.addEventListener("click", () => document.getElementById("deleteModal").style.display = "none");
    cancelDelModalBtn?.addEventListener("click", () => document.getElementById("deleteModal").style.display = "none");

    confirmDelBtn?.addEventListener("click", async () => {
        if (!AppState.deleteTargetId) return;
        try {
            const res = await fetch(`${API_BASE}/history/${AppState.deleteTargetId}`, { method: "DELETE" });
            if (res.ok) {
                showToast(`Analysis ${AppState.deleteTargetId} deleted.`, "success");
                document.getElementById("deleteModal").style.display = "none";
                loadHistoryTable();
                checkBackendHealth();
            }
        } catch (err) {
            showToast(`Delete failed: ${err.message}`, "error");
        }
    });
}

let historyCache = [];

async function loadHistoryTable() {
    const tbody = document.getElementById("historyTableBody");
    try {
        const res = await fetch(`${API_BASE}/history`);
        if (!res.ok) throw new Error("Failed to load history");
        historyCache = await res.json();
        renderHistoryRows(historyCache);
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center text-red">Error loading database records: ${err.message}</td></tr>`;
    }
}

function filterHistoryTable() {
    const query = document.getElementById("historySearchInput").value.toLowerCase();
    const filtered = historyCache.filter(item =>
        item.analysis_id.toLowerCase().includes(query) ||
        item.filename.toLowerCase().includes(query)
    );
    renderHistoryRows(filtered);
}

function renderHistoryRows(items) {
    const tbody = document.getElementById("historyTableBody");
    if (items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted">No historical records found.</td></tr>';
        return;
    }

    tbody.innerHTML = "";
    items.forEach(h => {
        const tr = document.createElement("tr");
        const timeStr = new Date(h.timestamp).toLocaleString();
        tr.innerHTML = `
            <td><strong>${h.analysis_id}</strong></td>
            <td>${timeStr}</td>
            <td>${h.source_type}</td>
            <td>${h.filename}</td>
            <td><strong class="${h.people_detected > 0 ? 'text-red' : ''}">${h.people_detected}</strong></td>
            <td>${h.best_confidence}%</td>
            <td>${h.thermal_palette}</td>
            <td><span class="badge badge-green">${h.status}</span></td>
            <td>
                <div style="display: flex; gap: 4px;">
                    <button class="btn btn-sm btn-secondary" onclick="viewHistoryDetail('${h.analysis_id}')">View</button>
                    <button class="btn btn-sm btn-outline" onclick="promptDeleteHistory('${h.analysis_id}')">Delete</button>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

window.viewHistoryDetail = async function (analysisId) {
    try {
        const res = await fetch(`${API_BASE}/history/${analysisId}`);
        if (!res.ok) throw new Error("Record not found");
        const data = await res.json();
        AppState.currentAnalysis = {
            analysis_id: data.analysis_id,
            filename: data.filename,
            source_type: data.source_type,
            people_detected: data.people_detected,
            best_confidence: data.best_confidence,
            visible_body_parts: data.visible_body_parts ? data.visible_body_parts.split(", ") : [],
            original_image_url: data.original_image_path,
            annotated_image_url: data.annotated_image_path,
            model_used: data.model_used,
            inference_time_ms: 0,
            location: {
                latitude: data.gps_latitude,
                longitude: data.gps_longitude,
                floor: data.floor_number,
                room: data.room_number,
                status: data.location_status
            },
            secondary_ai_analysis: {
                human_assessment: data.grok_assessment,
                visible_body_part: data.grok_body_part,
                evidence: data.grok_evidence,
                recommendation: data.grok_recommendation
            }
        };

        renderAnalysisResults(AppState.currentAnalysis);
        switchView("evidence");
    } catch (err) {
        showToast(`Failed to load analysis: ${err.message}`, "error");
    }
};

window.promptDeleteHistory = function (analysisId) {
    AppState.deleteTargetId = analysisId;
    document.getElementById("deleteTargetId").textContent = `ID: ${analysisId}`;
    document.getElementById("deleteModal").style.display = "flex";
};

// ==============================================================
// SETTINGS
// ==============================================================
function initSettingsHandlers() {
    const confSlider = document.getElementById("settingConfThreshold");
    const confLabel = document.getElementById("settingConfLabel");
    const saveBtn = document.getElementById("saveSettingsBtn");
    const testStreamBtn = document.getElementById("testStreamSettingsBtn");

    confSlider?.addEventListener("input", (e) => {
        confLabel.textContent = `${e.target.value}%`;
        AppState.confidenceThreshold = parseFloat(e.target.value) / 100.0;
        document.getElementById("uploadConfidenceSlider").value = e.target.value;
        document.getElementById("confValueLabel").textContent = `${e.target.value}%`;
    });

    const apiUrlInput = document.getElementById("settingApiUrl");
    if (apiUrlInput) {
        apiUrlInput.value = API_BASE;
        document.getElementById("saveApiUrlBtn")?.addEventListener("click", () => {
            const rawUrl = apiUrlInput.value.trim().replace(/\/$/, "");
            if (rawUrl) {
                API_BASE = rawUrl;
                localStorage.setItem("FIRE_FIGHTER_API_BASE", rawUrl);
                showToast(`Backend API target set to: ${rawUrl}`, "success");
                checkBackendHealth();
            }
        });
    }

    saveBtn?.addEventListener("click", () => {
        if (apiUrlInput && apiUrlInput.value.trim()) {
            const rawUrl = apiUrlInput.value.trim().replace(/\/$/, "");
            API_BASE = rawUrl;
            localStorage.setItem("FIRE_FIGHTER_API_BASE", rawUrl);
        }
        showToast("Settings successfully saved.", "success");
        checkBackendHealth();
    });

    testStreamBtn?.addEventListener("click", async () => {
        const url = document.getElementById("settingDroneUrl").value.trim();
        const resSpan = document.getElementById("streamTestResult");
        if (!url) {
            resSpan.textContent = "Drone stream not configured.";
            resSpan.className = "text-amber";
            return;
        }
        resSpan.textContent = "Checking...";
        try {
            const res = await fetch(`${API_BASE}/drone-config`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url, protocol: document.getElementById("settingProtocol").value })
            });
            const d = await res.json();
            resSpan.textContent = d.connection.message;
            resSpan.className = d.connection.connected ? "text-green" : "text-amber";
        } catch (e) {
            resSpan.textContent = e.message;
            resSpan.className = "text-red";
        }
    });

    // Print report
    document.getElementById("printReportBtn")?.addEventListener("click", () => window.print());

    // Downloads
    document.getElementById("downloadAnnotatedBtn")?.addEventListener("click", downloadCurrentAnnotated);
    document.getElementById("downloadAnnotatedBtn2")?.addEventListener("click", downloadCurrentAnnotated);
    document.getElementById("downloadOrigBtn")?.addEventListener("click", downloadCurrentOriginal);
}

function downloadCurrentAnnotated() {
    if (!AppState.currentAnalysis || !AppState.currentAnalysis.annotated_image_url) {
        showToast("No annotated image available.", "warning");
        return;
    }
    const a = document.createElement("a");
    a.href = `${API_BASE}${AppState.currentAnalysis.annotated_image_url}`;
    a.download = `annotated_${AppState.currentAnalysis.analysis_id}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function downloadCurrentOriginal() {
    if (!AppState.currentAnalysis || !AppState.currentAnalysis.original_image_url) {
        showToast("No original image available.", "warning");
        return;
    }
    const a = document.createElement("a");
    a.href = `${API_BASE}${AppState.currentAnalysis.original_image_url}`;
    a.download = `original_${AppState.currentAnalysis.analysis_id}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ==============================================================
// SAMPLE DATA BUTTONS FOR EASY TESTING
// ==============================================================
function initSampleDataButtons() {
    const handBtn = document.getElementById("loadSampleHandBtn");
    const personBtn = document.getElementById("loadSamplePersonBtn");

    async function loadSample(imageFilename) {
        try {
            showToast(`Loading ${imageFilename}...`, "info");
            const candidateUrls = [
                `images/${imageFilename}`,
                `results/${imageFilename}`,
                `/images/${imageFilename}`,
                `/results/${imageFilename}`,
                `../images/${imageFilename}`,
                `../results/${imageFilename}`,
                `${API_BASE}/results/${imageFilename}`
            ];
            let blob = null;
            for (const url of candidateUrls) {
                try {
                    const res = await fetch(url);
                    if (res.ok) {
                        blob = await res.blob();
                        break;
                    }
                } catch (e) {}
            }
            if (!blob) {
                throw new Error("Sample image not found on current server");
            }

            const file = new File([blob], imageFilename, { type: "image/jpeg" });
            handleSelectedImage(file);
            switchView("upload");
            showToast(`Loaded ${imageFilename} successfully! Click 'Run YOLO11 Detection'.`, "success");
        } catch (err) {
            showToast(`Could not load sample automatically: ${err.message}. Please use Browse Files.`, "warning");
        }
    }

    handBtn?.addEventListener("click", () => loadSample("sample_thermal_hand.jpg"));
    personBtn?.addEventListener("click", () => loadSample("sample_victim_person.jpg"));
}

// ==============================================================
// TOAST NOTIFICATION UTILITY
// ==============================================================
function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==============================================================
// TACTICAL MAP & INDOOR LOCALIZATION MODULE
// ==============================================================
let tacticalMap = null;
let droneMarker = null;
let victimBeaconMarker = null;
let searchPerimeterCircle = null;
let victimMarkersGroup = null;

const TacticalMap = {
    currentCoords: [17.3850, 78.4867],
    droneHomeCoords: [17.3850, 78.4867],
    activeFloor: "Floor 2 (Sector B)",
    activeRoom: "Room 204 (Open Workspace)",

    init() {
        this.initLeafletGpsMap();
        this.initFloorplan();
        this.initMapSwitchers();
        this.initManualLocationForm();
    },

    initLeafletGpsMap() {
        const mapContainer = document.getElementById("tacticalGpsMap");
        if (!mapContainer || typeof L === "undefined") return;

        try {
            tacticalMap = L.map("tacticalGpsMap", {
                center: this.currentCoords,
                zoom: 17,
                zoomControl: true
            });

            // --- FREE tile layers (no API key required) ---
            const esriSatellite = L.tileLayer(
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                {
                    attribution: 'Tiles &copy; <a href="https://www.esri.com">Esri</a> &mdash; Source: Esri, DigitalGlobe, GeoEye, i-cubed, USDA FSA, USGS, AEX, Getmapping, Aerogrid, IGN, IGP, swisstopo',
                    maxZoom: 19
                }
            );
            const osmStreet = L.tileLayer(
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }
            );
            esriSatellite.addTo(tacticalMap);

            // Layer switcher control
            L.control.layers(
                { "Satellite (Esri)": esriSatellite, "Street Map (OSM)": osmStreet },
                {},
                { position: "topright", collapsed: false }
            ).addTo(tacticalMap);

            victimMarkersGroup = L.layerGroup().addTo(tacticalMap);

            // Drone / Camera Home Marker
            const droneIcon = L.divIcon({
                className: "tactical-pulse-marker",
                html: '<div class="marker-drone-pin" style="background:#0ea5e9;border-radius:50%;width:28px;height:28px;border:3px solid #fff;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#fff;">CAM</div>',
                iconSize: [34, 34],
                iconAnchor: [17, 17]
            });
            droneMarker = L.marker(this.droneHomeCoords, { icon: droneIcon }).addTo(tacticalMap);
            droneMarker.bindPopup("<strong>DRONE / CAMERA SENSOR</strong><br>Aerial Search &amp; Rescue Unit<br>Status: Hovering &amp; Scanning");

            // Search Perimeter Circle (50m radius)
            searchPerimeterCircle = L.circle(this.droneHomeCoords, {
                radius: 50,
                color: "#38bdf8",
                fillColor: "#0284c7",
                fillOpacity: 0.12,
                weight: 1.5,
                dashArray: "4, 6"
            }).addTo(tacticalMap);

            // Victim Pinpoint Beacon
            const victimIcon = L.divIcon({
                className: "tactical-pulse-marker",
                html: '<div class="marker-victim-pin" style="background:#ef4444;border-radius:50%;width:28px;height:28px;border:3px solid #fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;">SOS</div>',
                iconSize: [34, 34],
                iconAnchor: [17, 17]
            });
            victimBeaconMarker = L.marker(this.currentCoords, { icon: victimIcon }).addTo(tacticalMap);
            victimBeaconMarker.bindPopup(
                `<strong>VICTIM CAPTURE LOCATION</strong><br>` +
                `Camera Lock: ${this.activeRoom}<br>` +
                `Coordinates: ${this.currentCoords[0].toFixed(5)}&deg; N, ${this.currentCoords[1].toFixed(5)}&deg; E<br>` +
                `Floor: ${this.activeFloor}`
            );

            // Map Click Event -> Pin exact camera capture location
            tacticalMap.on("click", (e) => {
                const { lat, lng } = e.latlng;
                this.setVictimCoordinates(lat, lng, true);
                showToast(`Camera location pinned: ${lat.toFixed(5)}, ${lng.toFixed(5)}`, "success");
            });

            // Auto-detect device / camera GPS location
            this.detectCameraGPS();

            // Load database markers
            this.loadDatabaseMarkers();

        } catch (mapErr) {
            console.warn("Leaflet map initialization glitch:", mapErr);
        }
    },

    // Detect real GPS location from browser (camera device geolocation)
    detectCameraGPS() {
        if (!navigator.geolocation) {
            showToast("Geolocation not supported by this browser.", "warning");
            return;
        }
        showToast("Detecting camera GPS location...", "info");
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                const accuracy = position.coords.accuracy;

                // Update drone home to actual detected location
                this.droneHomeCoords = [lat, lng];
                this.currentCoords   = [lat, lng];

                // Move markers
                if (droneMarker)  droneMarker.setLatLng([lat, lng]);
                if (victimBeaconMarker) victimBeaconMarker.setLatLng([lat, lng]);
                if (searchPerimeterCircle) searchPerimeterCircle.setLatLng([lat, lng]);

                // Pan map
                if (tacticalMap) tacticalMap.setView([lat, lng], 18, { animate: true });

                // Update UI fields
                document.getElementById("locLatDisplay").textContent = `${lat.toFixed(5)}° N`;
                document.getElementById("locLonDisplay").textContent = `${lng.toFixed(5)}° E`;
                document.getElementById("manualLatInput").value = lat.toFixed(5);
                document.getElementById("manualLonInput").value = lng.toFixed(5);
                const dashGps = document.getElementById("dashGpsCoords");
                if (dashGps) dashGps.textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;

                // Add accuracy circle
                L.circle([lat, lng], {
                    radius: accuracy,
                    color: "#facc15",
                    fillColor: "#fde047",
                    fillOpacity: 0.08,
                    weight: 1,
                    dashArray: "3, 5"
                }).addTo(tacticalMap).bindTooltip(`GPS Accuracy: ±${accuracy.toFixed(0)} m`);

                droneMarker.bindPopup(
                    `<strong>CAMERA / DEVICE LOCATION</strong><br>` +
                    `Lat: ${lat.toFixed(5)}&deg; N<br>` +
                    `Lon: ${lng.toFixed(5)}&deg; E<br>` +
                    `Accuracy: &plusmn;${accuracy.toFixed(0)} m`
                ).openPopup();

                showToast(`Camera GPS detected: ${lat.toFixed(4)}, ${lng.toFixed(4)} (±${accuracy.toFixed(0)}m)`, "success");
            },
            (err) => {
                let msg = "GPS unavailable";
                if (err.code === 1) msg = "Location access denied. Allow location in browser settings.";
                if (err.code === 2) msg = "GPS position unavailable.";
                if (err.code === 3) msg = "GPS request timed out.";
                showToast(msg, "warning");
                console.warn("Geolocation error:", err);
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    },

    setVictimCoordinates(lat, lng, isUserPin = false) {
        this.currentCoords = [lat, lng];
        if (victimBeaconMarker) {
            victimBeaconMarker.setLatLng([lat, lng]);
            victimBeaconMarker.getPopup()?.setContent(
                `<strong>VICTIM CAPTURE LOCATION</strong><br>` +
                `Latitude: ${lat.toFixed(5)}&deg; N<br>` +
                `Longitude: ${lng.toFixed(5)}&deg; E<br>` +
                `Floor: ${this.activeFloor} | Room: ${this.activeRoom}`
            );
            // Pan map to new pin
            if (tacticalMap) tacticalMap.setView([lat, lng], 18, { animate: true });
        }

        // Update UI displays
        document.getElementById("locLatDisplay").textContent = `${lat.toFixed(5)}° N`;
        document.getElementById("locLonDisplay").textContent = `${lng.toFixed(5)}° E`;
        document.getElementById("manualLatInput").value = lat.toFixed(5);
        document.getElementById("manualLonInput").value = lng.toFixed(5);

        if (AppState.currentAnalysis && AppState.currentAnalysis.location) {
            AppState.currentAnalysis.location.latitude = lat;
            AppState.currentAnalysis.location.longitude = lng;
        }

        const dashGps = document.getElementById("dashGpsCoords");
        if (dashGps) dashGps.textContent = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
    },

    centerOnVictim() {
        if (tacticalMap && this.currentCoords) {
            tacticalMap.setView(this.currentCoords, 18, { animate: true });
            if (victimBeaconMarker) victimBeaconMarker.openPopup();
        }
    },

    resetToDrone() {
        if (tacticalMap && this.droneHomeCoords) {
            tacticalMap.setView(this.droneHomeCoords, 17, { animate: true });
            if (droneMarker) droneMarker.openPopup();
        }
    },

    async loadDatabaseMarkers() {
        try {
            const res = await fetch(`${API_BASE}/locations/markers`);
            if (!res.ok) return;
            const data = await res.json();
            if (!data.markers || !victimMarkersGroup) return;

            victimMarkersGroup.clearLayers();

            data.markers.forEach(m => {
                if (m.gps_latitude && m.gps_longitude) {
                    const icon = L.divIcon({
                        className: "tactical-pulse-marker",
                        html: `<div class="marker-victim-pin" title="${m.detection_id}" style="background:#ef4444;border-radius:50%;width:22px;height:22px;border:2px solid #fff;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#fff;">SOS</div>`,
                        iconSize: [28, 28],
                        iconAnchor: [14, 14]
                    });
                    const marker = L.marker([m.gps_latitude, m.gps_longitude], { icon: icon });
                    marker.bindPopup(`
                        <div style="font-size: 12px; min-width: 140px;">
                            <strong>${m.detection_id} (${m.class_name.toUpperCase()})</strong><br>
                            Confidence: <strong class="text-green">${(m.confidence * 100).toFixed(1)}%</strong><br>
                            Status: <span class="badge badge-amber">${m.operator_status}</span><br>
                            Location: ${m.floor_number}, ${m.room_number}<br>
                            <small class="text-muted">${new Date(m.timestamp).toLocaleTimeString()}</small>
                        </div>
                    `);
                    victimMarkersGroup.addLayer(marker);
                }
            });
        } catch (e) {
            console.warn("Could not load map markers:", e);
        }
    },

    initFloorplan() {
        const floorBtns = document.querySelectorAll(".btn-floor");
        floorBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                floorBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const level = btn.getAttribute("data-floor");
                this.activeFloor = `Level ${level}`;
                document.getElementById("locFloorDisplay").textContent = `Floor ${level}`;
                document.getElementById("manualFloorInput").value = `Floor ${level}`;
                showToast(`Switched indoor view to: Floor ${level}`, "info");
            });
        });

        // Interactive room clicks
        const roomGroups = document.querySelectorAll(".room-group");
        roomGroups.forEach(grp => {
            grp.addEventListener("click", () => {
                const roomName = grp.getAttribute("data-room");
                const sector = grp.getAttribute("data-sector");
                this.activeRoom = roomName;

                // Move beacon to clicked room's center
                const rect = grp.querySelector("rect");
                if (rect) {
                    const rx = parseFloat(rect.getAttribute("x")) + parseFloat(rect.getAttribute("width")) / 2;
                    const ry = parseFloat(rect.getAttribute("y")) + parseFloat(rect.getAttribute("height")) / 2;

                    const beacon = document.getElementById("victimIndoorBeacon");
                    if (beacon) {
                        beacon.setAttribute("transform", `translate(${rx}, ${ry})`);
                        beacon.style.display = "block";
                    }

                    // Re-orient drone camera FOV cone toward the room
                    const fovCone = document.querySelector(".camera-fov-cone");
                    if (fovCone) {
                        const dx = rx - 400;
                        const dy = ry - 240;
                        fovCone.setAttribute("points", `0,0 ${dx - 20},${dy} ${dx + 20},${dy}`);
                    }
                }

                // Highlight room
                document.querySelectorAll(".room-rect").forEach(r => r.classList.remove("target-highlight"));
                rect?.classList.add("target-highlight");

                // Update text fields
                document.getElementById("locRoomDisplay").textContent = `${roomName} (${sector})`;
                document.getElementById("manualRoomInput").value = roomName;
                document.getElementById("dashRoom").textContent = roomName;

                showToast(`Victim pinned in: ${roomName}`, "success");
            });
        });
    },

    initMapSwitchers() {
        const gpsBtn = document.getElementById("mapModeGpsBtn");
        const floorBtn = document.getElementById("mapModeFloorBtn");
        const gpsContainer = document.getElementById("gpsMapContainer");
        const floorContainer = document.getElementById("indoorFloorContainer");

        gpsBtn?.addEventListener("click", () => {
            gpsBtn.classList.add("active");
            floorBtn?.classList.remove("active");
            gpsContainer.style.display = "block";
            floorContainer.style.display = "none";
            setTimeout(() => {
                if (tacticalMap) tacticalMap.invalidateSize();
            }, 100);
        });

        floorBtn?.addEventListener("click", () => {
            floorBtn.classList.add("active");
            gpsBtn?.classList.remove("active");
            gpsContainer.style.display = "none";
            floorContainer.style.display = "block";
        });

        document.getElementById("centerMapOnVictimBtn")?.addEventListener("click", () => this.centerOnVictim());
        document.getElementById("resetDroneMapBtn")?.addEventListener("click", () => this.resetToDrone());
    },

    initManualLocationForm() {
        const saveBtn = document.getElementById("saveManualLocBtn");
        saveBtn?.addEventListener("click", async () => {
            const floor = document.getElementById("manualFloorInput").value.trim() || "Floor 2 (Sector B)";
            const room = document.getElementById("manualRoomInput").value.trim() || "Room 204";
            const latStr = document.getElementById("manualLatInput").value.trim();
            const lonStr = document.getElementById("manualLonInput").value.trim();

            const lat = latStr ? parseFloat(latStr) : this.currentCoords[0];
            const lon = lonStr ? parseFloat(lonStr) : this.currentCoords[1];

            this.setVictimCoordinates(lat, lon);
            this.activeFloor = floor;
            this.activeRoom = room;

            document.getElementById("locFloorDisplay").textContent = floor;
            document.getElementById("locRoomDisplay").textContent = room;
            document.getElementById("locSourceDisplay").textContent = "Operator Verified Location";
            document.getElementById("dashFloor").textContent = floor;
            document.getElementById("dashRoom").textContent = room;

            const analysisId = AppState.currentAnalysis ? AppState.currentAnalysis.analysis_id : null;
            if (analysisId) {
                try {
                    const res = await fetch(`${API_BASE}/locations/update`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            analysis_id: analysisId,
                            latitude: lat,
                            longitude: lon,
                            floor: floor,
                            room: room,
                            status: "Operator Verified Coordinates"
                        })
                    });
                    if (res.ok) {
                        showToast("Location coordinates saved to database!", "success");
                        this.loadDatabaseMarkers();
                    }
                } catch (e) {
                    showToast(`Error saving location: ${e.message}`, "error");
                }
            } else {
                showToast("Coordinates attached to current session.", "success");
            }
        });
    }
};