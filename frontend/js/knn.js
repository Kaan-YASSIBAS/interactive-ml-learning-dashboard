const API_BASE_URL = "http://127.0.0.1:8000";

const kInput = document.getElementById("kInput");
const sepalLengthInput = document.getElementById("sepalLengthInput");
const sepalWidthInput = document.getElementById("sepalWidthInput");
const petalLengthInput = document.getElementById("petalLengthInput");
const petalWidthInput = document.getElementById("petalWidthInput");

const runButton = document.getElementById("runButton");
const errorBox = document.getElementById("errorBox");
const resultSection = document.getElementById("resultSection");

let knnChart = null;

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    resultSection.classList.add("hidden");
}

function hideError() {
    errorBox.classList.add("hidden");
}

function formatPercent(value) {
    return (value * 100).toFixed(2) + "%";
}

function updateBar(id, textId, value) {
    const bar = document.getElementById(id);
    const text = document.getElementById(textId);

    bar.style.width = formatPercent(value);
    text.textContent = formatPercent(value);
}

function createScatterPlot(points, samplePoint) {
    const setosa = [];
    const versicolor = [];
    const virginica = [];

    points.forEach((point) => {
        const chartPoint = {
            x: point.pc1,
            y: point.pc2
        };

        if (point.target_name === "setosa") {
            setosa.push(chartPoint);
        } else if (point.target_name === "versicolor") {
            versicolor.push(chartPoint);
        } else if (point.target_name === "virginica") {
            virginica.push(chartPoint);
        }
    });

    const sample = [
        {
            x: samplePoint.pc1,
            y: samplePoint.pc2
        }
    ];

    const ctx = document.getElementById("knnChart").getContext("2d");

    if (knnChart) {
        knnChart.destroy();
    }

    knnChart = new Chart(ctx, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Setosa",
                    data: setosa,
                    backgroundColor: "rgba(37, 99, 235, 0.7)"
                },
                {
                    label: "Versicolor",
                    data: versicolor,
                    backgroundColor: "rgba(16, 185, 129, 0.7)"
                },
                {
                    label: "Virginica",
                    data: virginica,
                    backgroundColor: "rgba(239, 68, 68, 0.7)"
                },
                {
                    label: "Girilen Yeni Örnek",
                    data: sample,
                    backgroundColor: "rgba(0, 0, 0, 1)",
                    pointRadius: 8,
                    pointHoverRadius: 10
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
                        text: "Principal Component 1"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Principal Component 2"
                    }
                }
            }
        }
    });
}

runButton.addEventListener("click", async function () {
    hideError();

    const k = Number(kInput.value);
    const sepalLength = Number(sepalLengthInput.value);
    const sepalWidth = Number(sepalWidthInput.value);
    const petalLength = Number(petalLengthInput.value);
    const petalWidth = Number(petalWidthInput.value);

    const queryParams = new URLSearchParams({
        k: k,
        sepal_length: sepalLength,
        sepal_width: sepalWidth,
        petal_length: petalLength,
        petal_width: petalWidth
    });

    try {
        const response = await fetch(`${API_BASE_URL}/knn?${queryParams.toString()}`);

        if (!response.ok) {
            throw new Error("KNN isteği başarısız oldu. K değeri 1 ile 15 arasında olmalıdır.");
        }

        const data = await response.json();

        document.getElementById("kValue").textContent = data.k;
        document.getElementById("accuracyValue").textContent = formatPercent(data.accuracy);
        document.getElementById("predictionValue").textContent = data.sample.prediction;

        updateBar("setosaBar", "setosaText", data.sample.probabilities.setosa || 0);
        updateBar("versicolorBar", "versicolorText", data.sample.probabilities.versicolor || 0);
        updateBar("virginicaBar", "virginicaText", data.sample.probabilities.virginica || 0);

        createScatterPlot(data.points, data.sample.pca_point);

        resultSection.classList.remove("hidden");

    } catch (error) {
        showError(error.message);
    }
});

runButton.click();