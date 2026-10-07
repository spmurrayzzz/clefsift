const SETTINGS_KEY = "systemone_settings";
const HISTORY_KEY = "systemone_history";
const MIN_WORDS = 10;
const VERDICT_TOP_MIN = 0.35;
const VERDICT_GAP_MIN = 0.1;
const COLORS = { ai: "#fb7185", "ai-assisted": "#fbbf24", mixed: "#c4b5fd", human: "#6ee7b7" };

const els = {
  setup: document.getElementById("setup"),
  setupButton: document.getElementById("setup-open-options"),
  textarea: document.getElementById("text"),
  count: document.getElementById("word-count"),
  scan: document.getElementById("scan"),
  result: document.getElementById("result"),
  history: document.getElementById("history"),
  historyEmpty: document.getElementById("history-empty"),
};

document.getElementById("open-options").addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});
els.setupButton.addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

function words(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

els.textarea.addEventListener("input", () => {
  const n = words(els.textarea.value);
  els.count.textContent = `${n} ${n === 1 ? "word" : "words"}`;
  els.scan.disabled = n < MIN_WORDS;
});

async function loadSettingsState() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  const settings = stored[SETTINGS_KEY] || {};
  const ready = !!settings.model;
  els.setup.classList.toggle("hidden", ready);
  els.result.classList.toggle("hidden", ready);
}

function renderResult(result) {
  els.result.classList.remove("hidden", "error");
  els.result.textContent = "";

  const entries = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);
  const topP = result.probabilities?.[result.choice] ?? 0;
  const runner = entries.find(([name]) => name !== result.choice)?.[1] ?? 0;
  const decisive = topP >= VERDICT_TOP_MIN && topP - runner >= VERDICT_GAP_MIN;

  const label = document.createElement("span");
  label.className = "result-label";
  label.style.color = decisive ? COLORS[result.choice] || "#cbd5e1" : "#cbd5e1";
  label.textContent = decisive
    ? result.choice.replace("-", " ").toUpperCase()
    : `UNCERTAIN (leaning ${result.choice.replace("-", " ")})`;
  els.result.appendChild(label);

  const conf = document.createElement("span");
  conf.className = "result-conf";
  conf.textContent = decisive
    ? `${Math.round(topP * 100)}% for this label`
    : `top label ${Math.round(topP * 100)}%, weak margin`;
  els.result.appendChild(conf);

  for (const [name, value] of entries) {
    const row = document.createElement("div");
    row.className = "prob-row";
    const nm = document.createElement("span");
    nm.className = "prob-name";
    nm.textContent = name;
    const bar = document.createElement("span");
    bar.className = "prob-bar";
    const fill = document.createElement("span");
    fill.className = "prob-fill";
    fill.style.width = `${Math.round((value || 0) * 100)}%`;
    fill.style.background = COLORS[name] || "#cbd5e1";
    bar.appendChild(fill);
    const pct = document.createElement("span");
    pct.className = "prob-pct";
    pct.textContent = `${Math.round((value || 0) * 100)}%`;
    row.append(nm, bar, pct);
    els.result.appendChild(row);
  }

  const model = document.createElement("div");
  model.className = "result-model";
  model.textContent = `model: ${result.model}${result.usage?.input_tokens != null ? ` · ${result.usage.input_tokens} input tokens` : ""}`;
  els.result.appendChild(model);
}

function renderError(message) {
  els.result.classList.remove("hidden");
  els.result.classList.add("error");
  els.result.textContent = message;
}

els.scan.addEventListener("click", () => {
  const text = els.textarea.value;
  if (words(text) < MIN_WORDS) return;
  els.scan.disabled = true;
  els.scan.textContent = "Checking...";
  els.result.classList.remove("hidden", "error");
  els.result.classList.add("checking");
  els.result.textContent = "Waiting for the decision model.";
  chrome.runtime.sendMessage({ type: "SCAN_TEXT", text }, (response) => {
    els.scan.disabled = false;
    els.scan.textContent = "Check text";
    els.result.classList.remove("checking");
    if (chrome.runtime.lastError) {
      renderError(chrome.runtime.lastError.message);
      return;
    }
    if (response?.ok) {
      renderResult(response.result);
      loadHistory();
    } else {
      renderError(response?.error || "Unknown error.");
    }
  });
});

async function loadHistory() {
  const stored = await chrome.storage.local.get(HISTORY_KEY);
  const list = stored[HISTORY_KEY] || [];
  els.history.textContent = "";
  els.historyEmpty.classList.toggle("hidden", list.length > 0);
  for (const entry of list.slice(0, 8)) {
    const li = document.createElement("li");
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.style.background = COLORS[entry.choice] || "#cbd5e1";
    chip.textContent = entry.choice.replace("-", " ");
    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = entry.snippet || "";
    snippet.title = entry.snippet || "";
    const when = document.createElement("span");
    when.className = "when";
    when.textContent = new Date(entry.ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    li.append(chip, snippet, when);
    els.history.appendChild(li);
  }
}

loadSettingsState();
loadHistory();
