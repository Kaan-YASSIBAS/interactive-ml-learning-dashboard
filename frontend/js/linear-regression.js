const API_BASE_URL = "http://127.0.0.1:8000";

const hoursInput = document.getElementById("hoursInput");
const predictButton = document.getElementById("predictButton");
const predictionResult = document.getElementById("predictionResult");
const errorBox = document.getElementById("errorBox");

let chart = null;

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    predictionResult.classList.add("hidden");
}

function hideError() {
    errorBox.classList.add("hidden");
}

function createChart(dataPoints) {
    const actualPoints = dataPoints.map((point) => ({
        x: point.hours_studied,
        y: point.actual_score
    }));

    const regressionLine = dataPoints.map((point) => ({
        x: point.hours_studied,
        y: point.predicted_score
    }));

    const ctx = document.getElementById("linearRegressionChart").getContext("2d");

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Gerçek Veriler",
                    data: actualPoints,
                    backgroundColor: "rgba(37, 99, 235, 0.8)"
                },
                {
                    label: "Regression Line",
                    data: regressionLine,
                    type: "line",
                    borderColor: "rgba(220, 38, 38, 0.9)",
                    backgroundColor: "rgba(220, 38, 38, 0.15)",
                    pointRadius: 3,
                    fill: false,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Çalışma Saati"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Sınav Notu"
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

async function loadLinearRegressionInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/linear-regression`);

        if (!response.ok) {
            throw new Error("Linear Regression bilgileri alınamadı.");
        }

        const data = await response.json();

        document.getElementById("weightValue").textContent = data.weight.toFixed(2);
        document.getElementById("biasValue").textContent = data.bias.toFixed(2);
        document.getElementById("maeValue").textContent = data.mae.toFixed(2);
        document.getElementById("r2Value").textContent = data.r2_score.toFixed(2);

        createChart(data.data_points);

    } catch (error) {
        showError(error.message);
    }
}

predictButton.addEventListener("click", async function () {
    hideError();

    const hoursStudied = Number(hoursInput.value);

    try {
        const response = await fetch(`${API_BASE_URL}/linear-regression/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                hours_studied: hoursStudied
            })
        });

        if (!response.ok) {
            throw new Error("Tahmin isteği başarısız oldu. Çalışma saati 0 ile 24 arasında olmalıdır.");
        }

        const data = await response.json();

        predictionResult.textContent = `${data.hours_studied} saat çalışma için tahmini sınav skoru: ${data.predicted_score.toFixed(2)}`;
        predictionResult.classList.remove("hidden");

    } catch (error) {
        showError(error.message);
    }
});

loadLinearRegressionInfo();