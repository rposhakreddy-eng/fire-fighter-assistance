const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const analyzeButton = document.getElementById("analyzeButton");

const status = document.getElementById("status");
const confidence = document.getElementById("confidence");
const bodyPart = document.getElementById("bodyPart");


imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    const imageURL = URL.createObjectURL(file);

    previewImage.src = imageURL;
    previewImage.style.display = "block";

    status.textContent = "Image selected.";
});


analyzeButton.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {
        status.textContent = "Please select a thermal image.";
        return;
    }

    status.textContent = "Analyzing image...";
    confidence.textContent = "";
    bodyPart.textContent = "";

    const formData = new FormData();

    formData.append("file", file);


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/analyze",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();
if (data.detected_object === "person") {
    status.textContent = "🟢 VICTIM DETECTED";
} else {
    status.textContent = "🔴 NO VICTIM DETECTED";
}

confidence.textContent =
    "Confidence: " + data.confidence + "%";

bodyPart.textContent =
    "Detected object: " + data.detected_object;

    } catch (error) {

        status.textContent =
            "Unable to connect to AI server.";

        console.error(error);
    }

});