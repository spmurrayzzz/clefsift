const SETTINGS_KEY = "systemone_settings";
const DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1";
const SAMPLE_TEXT =
  "The city council met on Tuesday to review the proposed budget for the next fiscal year. " +
  "Residents raised concerns about road maintenance and school funding. " +
  "After two hours of discussion, the council postponed the vote until next month, " +
  "when a revised proposal will be presented to the public for comment.";

const LINKEDIN_PATTERN = "https://*.linkedin.com/*";
const LINKEDIN_SCRIPT_ID = "clefsift-linkedin-feed";

const els = {
  form: document.getElementById("form"),
  baseUrl: document.getElementById("base-url"),
  apiKey: document.getElementById("api-key"),
  model: document.getElementById("model"),
  modelSelect: document.getElementById("model-select"),
  customModel: document.getElementById("custom-model-field"),
  modelStatus: document.getElementById("model-status"),
  connectionStatus: document.getElementById("connection-status"),
  unsavedStatus: document.getElementById("unsaved-status"),
  loadModels: document.getElementById("load-models"),
  save: document.getElementById("save"),
  test: document.getElementById("test"),
  status: document.getElementById("status"),
  linkedinToggle: document.getElementById("linkedin-feed-toggle"),
  dimToggle: document.getElementById("dim-ai-toggle"),
};

let saved = {};
let modelRequest;
let saving = false;
let testing = false;

function normalizeBase(value) {
  let text = (value || "").trim().replace(/\/+$/, "");
  if (text.endsWith("/systemone")) text = text.slice(0, -"/systemone".length);
  let url;
  try {
    url = new URL(text);
  } catch {
    throw new Error("Enter a valid server URL, such as http://127.0.0.1:8000/v1.");
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error("Server URL must start with https:// or http://.");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error("Use the server URL without login details, query parameters, or a fragment.");
  }
  return url;
}

function selectedModel() {
  return els.modelSelect.value === "__custom__" ? els.model.value.trim() : els.modelSelect.value;
}

function friendlyName(id) {
  return id.split(/[\\/]/).pop().replace(/\.gguf$/i, "").replace(/-/g, " ");
}

function setStatus(kind, message) {
  els.status.className = message ? `show ${kind}` : "";
  els.status.textContent = message;
}

function setModelStatus(state, message) {
  els.modelStatus.dataset.state = state;
  els.modelStatus.textContent = message;
}

function setConnectionStatus(state, message) {
  els.connectionStatus.dataset.state = state;
  els.connectionStatus.textContent = message;
}

function hasChanges() {
  let base = els.baseUrl.value.trim();
  try { base = normalizeBase(base).href.replace(/\/+$/, ""); } catch {}
  return base !== (saved.baseUrl || "").replace(/\/+$/, "") ||
    els.apiKey.value.trim() !== (saved.apiKey || "") || selectedModel() !== (saved.model || "");
}

function syncActions() {
  const changed = hasChanges();
  els.baseUrl.disabled = saving || testing;
  els.apiKey.disabled = saving || testing;
  els.model.disabled = saving || testing || !!modelRequest;
  els.modelSelect.disabled = saving || testing || !!modelRequest;
  els.loadModels.disabled = saving || testing || !!modelRequest;
  els.save.disabled = saving || testing || !!modelRequest;
  els.save.textContent = saving ? "Saving…" : selectedModel() ? "Save settings" : "Connect server";
  els.test.disabled = saving || testing || !!modelRequest || changed || !saved.model;
  els.test.textContent = testing ? "Testing…" : "Test connection";
  els.unsavedStatus.textContent = changed
    ? "Unsaved changes. Save before testing."
    : saved.model ? "All changes saved" : "Connect your server to choose a model.";
  els.customModel.hidden = els.modelSelect.value !== "__custom__";
}

function renderModels(models, selected) {
  els.modelSelect.replaceChildren(new Option("Choose a model", ""));
  els.modelSelect.options[0].disabled = true;
  for (const model of models) {
    els.modelSelect.add(new Option(model.label, model.id));
  }
  els.modelSelect.add(new Option("Custom model…", "__custom__"));
  els.modelSelect.value = models.some((model) => model.id === selected)
    ? selected : selected ? "__custom__" : "";
  if (selected) els.model.value = selected;
  syncActions();
}

async function loadSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  saved = stored[SETTINGS_KEY] || {};
  els.baseUrl.value = saved.baseUrl || DEFAULT_BASE_URL;
  els.apiKey.value = saved.apiKey || "";
  renderModels(saved.model ? [{ id: saved.model, label: friendlyName(saved.model) }] : [], saved.model || "");
  els.linkedinToggle.checked = !!saved.linkedinFeed;
  els.dimToggle.checked = !!saved.dimAiPosts;
  syncDimToggle();
  if (saved.baseUrl) await loadModels();
}

function syncDimToggle() {
  els.dimToggle.disabled = !els.linkedinToggle.checked || els.linkedinToggle.disabled;
}

async function persistSettings(changes) {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  const settings = { ...stored[SETTINGS_KEY], ...changes };
  await chrome.storage.local.set({ [SETTINGS_KEY]: settings });
  return settings;
}

async function loadModels() {
  modelRequest?.abort();
  const request = new AbortController();
  modelRequest = request;
  const timer = setTimeout(() => request.abort(), 10000);
  const selected = selectedModel();
  const custom = els.modelSelect.value === "__custom__" && selected !== saved.model;
  els.loadModels.disabled = true;
  els.loadModels.textContent = "Loading…";
  els.modelSelect.disabled = true;
  setModelStatus("loading", "Checking the models available from your server…");
  setConnectionStatus("checking", "Connecting");
  syncActions();
  try {
    const base = normalizeBase(els.baseUrl.value).href.replace(/\/+$/, "");
    const key = els.apiKey.value.trim();
    const response = await fetch(`${base}/models`, {
      headers: key ? { Authorization: `Bearer ${key}` } : {},
      signal: request.signal,
    });
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        throw new Error("The server rejected the API key. Check it and try again.");
      }
      if (response.status === 404) {
        throw new Error("This server does not list models. Choose Custom model and enter its model ID.");
      }
      throw new Error(`The server returned an error (HTTP ${response.status}). Try again when it is ready.`);
    }
    const data = await response.json();
    const entries = Array.isArray(data.models) && data.models.length ? data.models : data.data;
    const models = [];
    for (const entry of Array.isArray(entries) ? entries : []) {
      const id = entry.name || entry.id;
      if (typeof id !== "string" || !id.trim() || models.some((model) => model.id === id)) continue;
      models.push({ id, label: entry.displayName || entry.display_name || friendlyName(id) });
    }
    if (modelRequest !== request) return;
    const available = models.some((model) => model.id === selected);
    const next = available || custom ? selected : models.length === 1 ? models[0].id : "";
    renderModels(models, next);
    setConnectionStatus("idle", "Server reached");
    if (!models.length) {
      setModelStatus("empty", "The server returned no models. Load one on the server, then refresh, or enter a custom model ID.");
    } else if (selected && !available && !custom) {
      setModelStatus("warning", models.length === 1
        ? "The saved model is no longer listed. The available model is selected. Save to use it."
        : "The saved model is no longer listed. Choose an available model, then save.");
    } else {
      setModelStatus("ok", `${models.length} model${models.length === 1 ? "" : "s"} available from this server.`);
    }
  } catch (error) {
    if (modelRequest !== request) return;
    setConnectionStatus("error", "Needs attention");
    setModelStatus("error", error.name === "AbortError"
      ? "The server took too long to respond. Check that it is running, then refresh."
      : error instanceof TypeError
        ? "Could not reach this server. Check its address and that it is running. Save settings to allow access, then refresh."
        : error.message);
  } finally {
    clearTimeout(timer);
    if (modelRequest === request) {
      modelRequest = null;
      els.loadModels.disabled = false;
      els.loadModels.textContent = "Refresh models";
      els.modelSelect.disabled = false;
      syncActions();
    }
  }
}

for (const input of [els.baseUrl, els.apiKey]) {
  input.addEventListener("input", () => {
    modelRequest?.abort();
    modelRequest = null;
    els.loadModels.disabled = false;
    els.loadModels.textContent = "Refresh models";
    els.modelSelect.disabled = false;
    renderModels([], selectedModel());
    setConnectionStatus("idle", "Not tested");
    setModelStatus("empty", "Refresh models to check this connection.");
    setStatus("", "");
  });
}

for (const input of [els.modelSelect, els.model]) {
  input.addEventListener("input", () => {
    setConnectionStatus("idle", "Not tested");
    setModelStatus("empty", els.modelSelect.value === "__custom__"
      ? "Enter the exact model ID accepted by your server."
      : hasChanges() ? "Save settings to use this model." : "This model is saved for your connection.");
    setStatus("", "");
    syncActions();
  });
}

els.loadModels.addEventListener("click", loadModels);

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (saving || testing || modelRequest) return;
  saving = true;
  syncActions();
  try {
    const url = normalizeBase(els.baseUrl.value);
    const apiKey = els.apiKey.value.trim();
    const model = selectedModel();
    const granted = await chrome.permissions.request({ origins: [`${url.origin}/*`] });
    if (!granted) {
      setStatus("err", "Server access was not granted. Your saved settings have not changed.");
      return;
    }
    if (!model) {
      await loadModels();
      if (selectedModel()) setStatus("info", "Server connected. Save settings to use the selected model.");
      return;
    }
    saved = await persistSettings({ baseUrl: url.href.replace(/\/+$/, ""), apiKey, model });
    els.baseUrl.value = saved.baseUrl;
    setModelStatus("empty", "This model is saved for your connection.");
    setConnectionStatus("idle", "Not tested");
    setStatus("ok", "Settings saved. Reload open LinkedIn tabs to use this connection.");
    syncActions();
  } catch (error) {
    setStatus("err", error.message);
  } finally {
    saving = false;
    syncActions();
  }
});

els.linkedinToggle.addEventListener("change", async () => {
  const enabled = els.linkedinToggle.checked;
  if (enabled && !saved.model) {
    els.linkedinToggle.checked = false;
    syncDimToggle();
    setStatus("err", "Save a model before turning on automatic checks.");
    return;
  }
  els.linkedinToggle.disabled = true;
  syncDimToggle();
  try {
    if (enabled) {
      const granted = await chrome.permissions.request({ origins: [LINKEDIN_PATTERN] });
      if (!granted) throw new Error("LinkedIn access was not granted. Automatic checks remain off.");
      try {
        await chrome.scripting.registerContentScripts([{
          id: LINKEDIN_SCRIPT_ID,
          matches: [LINKEDIN_PATTERN],
          js: ["content/linkedin-feed.js"],
          runAt: "document_idle",
          persistAcrossSessions: true,
        }]);
      } catch (error) {
        if (!/duplicate/i.test(error.message)) throw error;
      }
    } else {
      try { await chrome.scripting.unregisterContentScripts({ ids: [LINKEDIN_SCRIPT_ID] }); } catch {}
    }
    await persistSettings({ linkedinFeed: enabled });
    saved.linkedinFeed = enabled;
    if (!enabled) {
      try { await chrome.permissions.remove({ origins: [LINKEDIN_PATTERN] }); } catch {}
    }
    setStatus("ok", `Automatic checks are ${enabled ? "on" : "off"}. Reload open LinkedIn tabs to apply this change.`);
  } catch (error) {
    els.linkedinToggle.checked = !!saved.linkedinFeed;
    setStatus("err", error.message);
  } finally {
    els.linkedinToggle.disabled = false;
    syncDimToggle();
  }
});

els.dimToggle.addEventListener("change", async () => {
  const enabled = els.dimToggle.checked;
  els.dimToggle.disabled = true;
  try {
    await persistSettings({ dimAiPosts: enabled });
    saved.dimAiPosts = enabled;
    setStatus("ok", enabled ? "Dimming is on. Checked posts update immediately." : "Dimming is off. Checked posts update immediately.");
  } catch (error) {
    els.dimToggle.checked = !!saved.dimAiPosts;
    setStatus("err", error.message);
  } finally {
    syncDimToggle();
  }
});

els.test.addEventListener("click", async () => {
  if (hasChanges() || !saved.model || testing) return;
  testing = true;
  syncActions();
  setConnectionStatus("checking", "Testing");
  setStatus("info", "Checking the saved model with a short sample…");
  try {
    const response = await chrome.runtime.sendMessage({ type: "SCAN_TEXT", text: SAMPLE_TEXT });
    if (!response?.ok) throw new Error(response?.error || "The model did not return a result.");
    if (hasChanges()) return;
    setConnectionStatus("ok", "Connected");
    setStatus("ok", `Connection verified. ${friendlyName(response.result.model || saved.model)} returned a result.`);
  } catch (error) {
    if (hasChanges()) return;
    setConnectionStatus("error", "Test failed");
    setStatus("err", error.message);
  } finally {
    testing = false;
    syncActions();
  }
});

loadSettings().catch((error) => setStatus("err", `Could not load settings. ${error.message}`));
