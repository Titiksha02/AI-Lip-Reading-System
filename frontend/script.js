console.log("script.js loaded");

const videoInput = document.getElementById("videoInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

// --------------------
// Preview Selected Video
// --------------------
videoInput.addEventListener("change", function () {
  const file = this.files[0];

  if (!file) return;

  result.innerHTML = "✅ Video selected. Click Predict Speech.";

  // MPG files cannot be previewed in most browsers
  if (file.name.toLowerCase().endsWith(".mpg")) {
    preview.removeAttribute("src");
    preview.load();

    result.innerHTML +=
      "<br><br>⚠️ MPG preview is not supported by browsers. After prediction, the converted MP4 will be shown.";

    return;
  }

  const url = URL.createObjectURL(file);

  preview.src = url;
  preview.style.display = "block";
  preview.load();
});

// --------------------
// Predict
// --------------------
async function predictVideo() {
  console.log("Predict button clicked");

  const file = videoInput.files[0];

  if (!file) {
    alert("Please select a video first.");

    return;
  }

  const formData = new FormData();

  formData.append("video", file);

  result.innerHTML = `
        <div style="padding:20px;">
            ⏳ Processing Video...
        </div>
    `;

  try {
    const response = await fetch("/predict", {
      method: "POST",

      body: formData,
    });

    if (!response.ok) {
      throw new Error("Server Error : " + response.status);
    }

    const data = await response.json();

    console.log(data);

    // -----------------------
    // Show converted MP4
    // -----------------------
    if (data.video) {
      preview.src = data.video + "?t=" + new Date().getTime();

      preview.style.display = "block";

      preview.load();
    }

    result.innerHTML = `
            <h3>Prediction Result</h3>

            <p>${data.prediction}</p>
        `;
  } catch (err) {
    console.error(err);

    result.innerHTML = `
            <h3 style="color:red;">
                Prediction Failed
            </h3>

            <p>${err.message}</p>
        `;
  }
}
