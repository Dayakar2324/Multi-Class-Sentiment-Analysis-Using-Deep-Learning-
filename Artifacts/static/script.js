const API_URL = "/prediction";

const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const analyzeButton = document.getElementById("analyze-button");
const buttonLabel = document.getElementById("button-label");
const clearButton = document.getElementById("clear-button");
const errorMessage = document.getElementById("error-message");

const emptyState = document.getElementById("empty-state");
const predictionContent = document.getElementById("prediction-content");
const resultState = document.getElementById("result-state");
const predictedEmotion = document.getElementById("predicted-emotion");
const emotionEmoji = document.getElementById("emotion-emoji");
const confidenceValue = document.getElementById("confidence-value");
const probabilityList = document.getElementById("probability-list");

const EMOTION_EMOJIS = {
  sadness: "😢",
  joy: "😊",
  love: "❤️",
  anger: "😡",
  fear: "😨",
  surprise: "😲"
};

textInput.addEventListener("input", () => {
  charCount.textContent = `${textInput.value.length} / 2000`;
  hideError();
});

clearButton.addEventListener("click", () => {
  textInput.value = "";
  charCount.textContent = "0 / 2000";
  hideError();
  textInput.focus();
});

analyzeButton.addEventListener("click", analyzeEmotion);

textInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    analyzeEmotion();
  }
});

async function analyzeEmotion() {
  const text = textInput.value.trim();

  if (!text) {
    showError("Please enter some text before running the analysis.");
    textInput.focus();
    return;
  }

  setLoading(true);
  hideError();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = typeof data.detail === "string"
        ? data.detail
        : `The request failed with status ${response.status}.`;
      throw new Error(detail);
    }

    renderPrediction(data);
  } catch (error) {
    showError(
      error.message || "Unable to reach the prediction service. Please try again."
    );
  } finally {
    setLoading(false);
  }
}

function renderPrediction(data) {
  const emotion = String(data.predicted_emotion || "").toLowerCase();
  const confidence = Number(data.confidence);
  const probabilities = data.all_probabilities || {};

  if (!emotion || !Number.isFinite(confidence)) {
    showError("The API returned an unexpected response. Please check the backend response schema.");
    return;
  }

  predictedEmotion.textContent = emotion;
  emotionEmoji.textContent = data.emoji || EMOTION_EMOJIS[emotion] || "•";
  confidenceValue.textContent = `Confidence: ${formatPercent(confidence)}`;

  const sortedProbabilities = Object.entries(probabilities)
    .map(([label, value]) => [label, Number(value)])
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => b[1] - a[1]);

  probabilityList.replaceChildren();

  sortedProbabilities.forEach(([label, value], index) => {
    const row = document.createElement("div");
    row.className = `probability-row${index === 0 ? " is-top" : ""}`;

    const labelElement = document.createElement("span");
    labelElement.className = "probability-label";
    labelElement.textContent = label;

    const track = document.createElement("div");
    track.className = "bar-track";
    track.setAttribute("aria-label", `${label}: ${formatPercent(value)}`);

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(value * 100, 100))}%`;
    track.appendChild(fill);

    const percentage = document.createElement("span");
    percentage.className = "probability-value";
    percentage.textContent = formatPercent(value);

    row.append(labelElement, track, percentage);
    probabilityList.appendChild(row);
  });

  emptyState.hidden = true;
  predictionContent.hidden = false;
  resultState.textContent = "Analysis complete";
}

function formatPercent(value) {
  const normalized = value <= 1 ? value * 100 : value;
  return `${normalized.toFixed(1)}%`;
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  buttonLabel.textContent = isLoading ? "Analyzing..." : "Analyze emotion";
  resultState.textContent = isLoading ? "Processing input" : "Ready";
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function hideError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}
