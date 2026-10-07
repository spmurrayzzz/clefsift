const SETTINGS_KEY = "systemone_settings";
const HISTORY_KEY = "systemone_history";
const DEFAULT_SETTINGS = { baseUrl: "http://127.0.0.1:8000/v1", apiKey: "" };
const MIN_WORDS = 10;
const MAX_CHARS = 50000;
const HISTORY_LIMIT = 20;
const MENU_SELECTION = "clefsift-check-selection";
const MENU_PAGE = "clefsift-check-page";

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function getSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  return { ...DEFAULT_SETTINGS, ...(stored[SETTINGS_KEY] || {}) };
}

function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function buildRequestBody(model, text) {
  return {
    model,
    state: text.slice(0, MAX_CHARS),
    questions: {
      classification: {
        type: "choice",
        instructions:
          "Judge only the writing style, not the topic or quality. Human writing tends to show irregular grammar, typos, uneven rhythm, slang, humor, and lived personal detail. AI-generated writing tends to show flawless grammar, uniformly balanced sentences, generic abstraction, hedging, and formulaic transitions.",
        criteria: {
          human:
            "Irregular grammar or typos, colloquial voice, slang or humor, uneven sentence rhythm, concrete personal detail from lived experience",
          "ai-assisted":
            "Mostly uniform, polished, machine-like prose, but with some personal specifics or human irregularities breaking through",
          mixed: "Contains some clearly human-written passages mixed with some clearly machine-generated passages",
          ai: "Flawless uniform prose, balanced clause structures, generic abstract content, hedging language, formulaic transitions, no personal irregularities",
        },
      },
    },
  };
}

async function responseError(response) {
  const retryAfter = response.headers.get("Retry-After");
  if (response.status === 401 || response.status === 403) {
    return "Unauthorized. Check the API key in the extension options.";
  }
  if (response.status === 429) {
    return `Rate limited${retryAfter ? `. Retry in ${retryAfter}s` : ""}.`;
  }
  let detail = "";
  try {
    const body = await response.json();
    if (body?.detail) detail = JSON.stringify(body.detail);
    else if (body?.error) detail = JSON.stringify(body.error);
  } catch {}
  return `HTTP ${response.status}${detail ? `. ${detail.slice(0, 300)}` : ""}`;
}

async function classify(text, settings) {
  const headers = {
    "Content-Type": "application/json",
    "Idempotency-Key": crypto.randomUUID(),
  };
  if (settings.apiKey) headers.Authorization = `Bearer ${settings.apiKey}`;
  const response = await fetch(`${settings.baseUrl}/systemone`, {
    method: "POST",
    headers,
    body: JSON.stringify(buildRequestBody(settings.model, text)),
  });
  if (!response.ok) throw new HttpError(await responseError(response), response.status);
  const data = await response.json();
  const answer = data?.answers?.classification;
  if (answer?.type !== "choice" || !answer.choice) {
    throw new Error("Unexpected response. Missing answers.classification choice.");
  }
  return {
    model: data.model || settings.model,
    choice: answer.choice,
    probabilities: answer.probabilities || {},
    confidence: answer.confidence ?? null,
    usage: data.usage || null,
  };
}

async function saveHistory(result, pageUrl, snippet) {
  const stored = await chrome.storage.local.get(HISTORY_KEY);
  const entry = { ts: Date.now(), url: pageUrl || "", snippet, ...result };
  const list = [entry, ...(stored[HISTORY_KEY] || [])].slice(0, HISTORY_LIMIT);
  await chrome.storage.local.set({ [HISTORY_KEY]: list });
}

async function injectToast(tabId, payload) {
  await chrome.scripting.executeScript({
    target: { tabId },
    func: pageToast,
    args: [payload],
  });
}

function pageToast(payload) {
  const COLORS = { ai: "#fb7185", "ai-assisted": "#fbbf24", mixed: "#c4b5fd", human: "#6ee7b7" };
  const VERDICT_TOP_MIN = 0.35;
  const VERDICT_GAP_MIN = 0.1;
  document.getElementById("clefsift-toast")?.remove();
  const toast = document.createElement("div");
  toast.id = "clefsift-toast";
  toast.style.cssText =
    "position:fixed;top:16px;right:16px;z-index:2147483647;max-width:360px;box-sizing:border-box;padding:14px 16px;border:1px solid #3a4149;border-radius:12px;background:#1d2226;color:#f0f1f2;color-scheme:dark;font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;box-shadow:0 10px 30px rgba(0,0,0,.4);cursor:pointer";

  const title = document.createElement("div");
  if (payload.phase === "loading") {
    title.style.cssText = "font-weight:600";
    title.textContent = "Checking for AI text...";
    const dots = document.createElement("span");
    dots.textContent = " ..";
    dots.style.cssText = "color:#a5b4fc";
    title.appendChild(dots);
  } else if (payload.phase === "result") {
    const entries = Object.entries(payload.probabilities || {}).sort(
      (a, b) => b[1] - a[1],
    );
    const topP = payload.probabilities?.[payload.choice] ?? 0;
    const runner = entries.find(([name]) => name !== payload.choice)?.[1] ?? 0;
    const decisive = topP >= VERDICT_TOP_MIN && topP - runner >= VERDICT_GAP_MIN;
    title.style.cssText = "font-weight:700;font-size:14px";
    title.style.color = decisive
      ? COLORS[payload.choice] || "#cbd5e1"
      : "#cbd5e1";
    title.textContent = decisive
      ? payload.choice.replace("-", " ").toUpperCase()
      : `UNCERTAIN (leaning ${payload.choice.replace("-", " ")})`;
  } else {
    title.style.cssText = "font-weight:700;color:#f87171";
    title.textContent = "Check failed";
  }
  toast.appendChild(title);

  const body = document.createElement("div");
  if (payload.phase === "result") {
    const entries = Object.entries(payload.probabilities || {}).sort((a, b) => b[1] - a[1]);
    for (const [name, value] of entries) {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:6px";
      const nm = document.createElement("span");
      nm.textContent = name;
      nm.style.cssText = "width:86px;flex:none;color:#cbd5e1;text-transform:capitalize";
      const bar = document.createElement("span");
      bar.style.cssText = "flex:1;display:block;height:6px;border-radius:3px;background:#343c44;overflow:hidden";
      const fill = document.createElement("span");
      fill.style.cssText = `display:block;height:6px;width:${Math.round((value || 0) * 100)}%;background:${COLORS[name] || "#cbd5e1"}`;
      bar.appendChild(fill);
      const pct = document.createElement("span");
      pct.textContent = `${Math.round((value || 0) * 100)}%`;
      pct.style.cssText = "width:38px;flex:none;text-align:right;color:#e5e7eb";
      row.append(nm, bar, pct);
      body.appendChild(row);
    }
    if (payload.model) {
      const m = document.createElement("div");
      m.textContent = `model: ${payload.model}`;
      m.style.cssText = "margin-top:8px;color:#b0b8c1;font-size:11px;overflow-wrap:anywhere";
      body.appendChild(m);
    }
  } else if (payload.phase === "error") {
    body.textContent = payload.message;
    body.style.cssText = "margin-top:6px;color:#e5e7eb";
  } else {
    body.textContent = "Waiting for the decision model.";
    body.style.cssText = "margin-top:6px;color:#b0b8c1";
  }
  toast.appendChild(body);

  toast.addEventListener("click", () => toast.remove());
  document.documentElement.appendChild(toast);
  setTimeout(() => toast.remove(), payload.phase === "result" ? 12000 : 6000);
}

const inFlightTabs = new Set();

async function scanText(rawText, { tabId = null, pageUrl = "", saveHistory: shouldSaveHistory = true } = {}) {
  const run = async () => {
    const trimmed = (rawText || "").trim();
    if (!trimmed) throw new Error("No text found to check.");
    const words = wordCount(trimmed);
    if (words < MIN_WORDS) {
      throw new Error(`Too short to check: ${words} words, minimum ${MIN_WORDS}.`);
    }
    const settings = await getSettings();
    if (!settings.model) throw new Error("No model set. Open the extension options and load models.");
    const result = await classify(trimmed, settings);
    if (shouldSaveHistory) await saveHistory(result, pageUrl, trimmed.slice(0, 140));
    return result;
  };

  if (tabId == null) return run();
  if (inFlightTabs.has(tabId)) throw new Error("A check is already running on this tab.");
  inFlightTabs.add(tabId);
  try {
    await injectToast(tabId, { phase: "loading" });
  } catch {}
  try {
    const result = await run();
    try {
      await injectToast(tabId, { phase: "result", ...result });
    } catch {}
    return result;
  } catch (error) {
    try {
      await injectToast(tabId, { phase: "error", message: error.message });
    } catch {}
    throw error;
  } finally {
    inFlightTabs.delete(tabId);
  }
}

async function restoreLinkedinFeed() {
  const id = "clefsift-linkedin-feed";
  const legacyId = "systemone-linkedin-feed";
  const scripts = await chrome.scripting.getRegisteredContentScripts({ ids: [id, legacyId] });
  if (scripts.some((script) => script.id === legacyId)) {
    await chrome.scripting.unregisterContentScripts({ ids: [legacyId] });
  }
  const settings = await getSettings();
  if (!settings.linkedinFeed) return;
  const origins = ["https://*.linkedin.com/*"];
  if (!(await chrome.permissions.contains({ origins }))) return;
  if (scripts.some((script) => script.id === id)) return;
  await chrome.scripting.registerContentScripts([
    {
      id,
      matches: origins,
      js: ["content/linkedin-feed.js"],
      runAt: "document_idle",
      persistAcrossSessions: true,
    },
  ]);
}

function restoreSavedFeed() {
  restoreLinkedinFeed().catch((error) => {
    console.warn("Could not restore LinkedIn feed scanning:", error.message);
  });
}

chrome.runtime.onStartup.addListener(restoreSavedFeed);

chrome.runtime.onInstalled.addListener(() => {
  restoreSavedFeed();
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: MENU_SELECTION,
      title: "Check for AI text",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: MENU_PAGE,
      title: "Check this page for AI text",
      contexts: ["page"],
    });
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  if (info.menuItemId !== MENU_SELECTION && info.menuItemId !== MENU_PAGE) return;
  try {
    let text = "";
    if (info.menuItemId === MENU_SELECTION) {
      try {
        const [{ result }] = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => window.getSelection().toString(),
        });
        text = result || info.selectionText || "";
      } catch {
        text = info.selectionText || "";
      }
    } else {
      const [{ result }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => document.body.innerText,
      });
      text = result || "";
    }
    await scanText(text, { tabId: tab.id, pageUrl: tab.url || "" });
  } catch (error) {
    console.warn("ClefSift scan failed:", error.message);
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "SCAN_TEXT") {
    scanText(message.text, {
      pageUrl: message.pageUrl || "",
      saveHistory: message.source !== "linkedin-feed",
    })
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }
});
