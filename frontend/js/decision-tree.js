const API_BASE_URL = "http://127.0.0.1:8000";

mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    flowchart: {
        curve: "basis"
    }
});

const maxDepthInput = document.getElementById("maxDepthInput");
const sepalLengthInput = document.getElementById("sepalLengthInput");
const sepalWidthInput = document.getElementById("sepalWidthInput");
const petalLengthInput = document.getElementById("petalLengthInput");
const petalWidthInput = document.getElementById("petalWidthInput");

const runButton = document.getElementById("runButton");
const errorBox = document.getElementById("errorBox");
const resultSection = document.getElementById("resultSection");
const rulesTableBody = document.getElementById("rulesTableBody");

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

function updateRulesTable(rules) {
    rulesTableBody.innerHTML = "";

    rules.forEach((item) => {
        const row = document.createElement("tr");

        row.innerHTML = `
            <td class="rules-depth">${item.depth}</td>
            <td class="rules-type">${item.type}</td>
            <td class="rules-rule" style="padding-left: ${12 + item.depth * 18}px;">
                ${item.rule}
            </td>
        `;

        rulesTableBody.appendChild(row);
    });
}

function sanitizeNodeText(text) {
    return text
        .replaceAll("(", "")
        .replaceAll(")", "")
        .replaceAll("[", "")
        .replaceAll("]", "")
        .replaceAll("<=", "≤");
}

function buildMermaidDiagram(treeStructure) {
    let nodeCounter = 0;
    const lines = ["graph TD"];

    function createNode(node) {
        const currentId = `N${nodeCounter}`;
        nodeCounter += 1;

        if (node.type === "decision") {
            const ruleText = sanitizeNodeText(node.rule);
            lines.push(`${currentId}{"${ruleText}?"}`);

            const leftId = createNode(node.left);
            const rightId = createNode(node.right);

            lines.push(`${currentId} -->|Evet| ${leftId}`);
            lines.push(`${currentId} -->|Hayır| ${rightId}`);
        } else {
            const predictionText = sanitizeNodeText(node.prediction);
            lines.push(`${currentId}["Predict: ${predictionText}"]`);
        }

        return currentId;
    }

    createNode(treeStructure);

    return lines.join("\n");
}

async function renderTreeDiagram(treeStructure) {
    const diagramContainer = document.getElementById("treeDiagram");
    const diagram = buildMermaidDiagram(treeStructure);

    diagramContainer.removeAttribute("data-processed");
    diagramContainer.innerHTML = diagram;

    await mermaid.run({
        nodes: [diagramContainer]
    });
}

runButton.addEventListener("click", async function () {
    hideError();

    const maxDepth = Number(maxDepthInput.value);
    const sepalLength = Number(sepalLengthInput.value);
    const sepalWidth = Number(sepalWidthInput.value);
    const petalLength = Number(petalLengthInput.value);
    const petalWidth = Number(petalWidthInput.value);

    const queryParams = new URLSearchParams({
        max_depth: maxDepth,
        sepal_length: sepalLength,
        sepal_width: sepalWidth,
        petal_length: petalLength,
        petal_width: petalWidth
    });

    try {
        const response = await fetch(`${API_BASE_URL}/decision-tree?${queryParams.toString()}`);

        if (!response.ok) {
            throw new Error("Decision Tree isteği başarısız oldu. Max depth 1 ile 10 arasında olmalıdır.");
        }

        const data = await response.json();

        document.getElementById("depthValue").textContent = data.max_depth;
        document.getElementById("trainAccuracyValue").textContent = formatPercent(data.train_accuracy);
        document.getElementById("testAccuracyValue").textContent = formatPercent(data.test_accuracy);
        document.getElementById("predictionValue").textContent = data.sample.prediction;

        updateBar("setosaBar", "setosaText", data.sample.probabilities.setosa || 0);
        updateBar("versicolorBar", "versicolorText", data.sample.probabilities.versicolor || 0);
        updateBar("virginicaBar", "virginicaText", data.sample.probabilities.virginica || 0);

        updateRulesTable(data.tree_rules);
        await renderTreeDiagram(data.tree_structure);

        resultSection.classList.remove("hidden");

    } catch (error) {
        showError(error.message);
    }
});

runButton.click();