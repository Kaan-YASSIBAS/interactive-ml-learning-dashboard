const API_BASE_URL = "http://127.0.0.1:8000";

const messageInput = document.getElementById("messageInput");
const predictButton = document.getElementById("predictButton");
const predictionResult = document.getElementById("predictionResult");
const errorBox = document.getElementById("errorBox");

const probabilitySection = document.getElementById("probabilitySection");
const hamProbabilityText = document.getElementById("hamProbabilityText");
const spamProbabilityText = document.getElementById("spamProbabilityText");
const hamBar = document.getElementById("hamBar");
const spamBar = document.getElementById("spamBar");

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    predictionResult.classList.add("hidden");
    probabilitySection.classList.add("hidden");
}

function hideError() {
    errorBox.classList.add("hidden");
}

function formatPercent(value) {
    return (value * 100).toFixed(2) + "%";
}

function updateProbabilityBars(probabilities) {
    const hamValue = probabilities.ham || 0;
    const spamValue = probabilities.spam || 0;

    hamProbabilityText.textContent = formatPercent(hamValue);
    spamProbabilityText.textContent = formatPercent(spamValue);

    hamBar.style.width = formatPercent(hamValue);
    spamBar.style.width = formatPercent(spamValue);

    probabilitySection.classList.remove("hidden");
}

predictButton.addEventListener("click", async function () {
    hideError();

    const message = messageInput.value.trim();

    if (!message) {
        showError("Mesaj boş olamaz.");
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/logistic-regression/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        if (!response.ok) {
            throw new Error("Tahmin isteği başarısız oldu.");
        }

        const data = await response.json();

        predictionResult.textContent = `Tahmin sonucu: ${data.prediction.toUpperCase()}`;
        predictionResult.classList.remove("hidden");

        updateProbabilityBars(data.probabilities);

    } catch (error) {
        showError(error.message);
    }
});