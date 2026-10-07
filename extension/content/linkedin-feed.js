(() => {
  const SETTINGS_KEY = "systemone_settings";
  const FEED_MIN_WORDS = 50;
  const SCAN_TIMEOUT_MS = 120000;
  const MAX_QUEUE = 40;
  const DEBOUNCE_MS = 400;
  const STYLE_ID = "clefsift-feed-loading-style";
  const VERDICT_TOP_MIN = 0.35;
  const VERDICT_GAP_MIN = 0.1;

  const COLORS = { ai: "#fb7185", "ai-assisted": "#fbbf24", mixed: "#c4b5fd", human: "#6ee7b7" };
  const BACKGROUNDS = { ai: "#3b2029", "ai-assisted": "#382d1d", mixed: "#30263f", human: "#1c352d" };
  const LABELS = {
    ai: "AI",
    "ai-assisted": "AI-ASSISTED",
    mixed: "MIXED",
    human: "HUMAN",
  };
  const DIM_CHOICES = new Set(["ai", "ai-assisted"]);

  const POST_CONTAINERS = ".fie-impression-container, div:has(> h2)";
  const POST_TEXT =
    '[data-testid="expandable-text-box"], .update-components-update-v2__commentary';
  const INSERTION_TARGETS = [
    ".update-components-actor__title",
    'a[href*="linkedin.com/in/"] > div > div, a[href^="/in/"] > div > div, a[href*="linkedin.com/company/"] > div > div, a[href^="/company/"] > div > div',
    "div[aria-label*=' Profile']",
  ];

  let enabled = false;
  let dimEnabled = false;
  let observer = null;
  let debounceTimer = null;
  let activeKey = null;
  const cache = new Map();
  const pending = new Set();
  const queue = [];

  function hash(text) {
    let h = 0;
    for (let i = 0; i < text.length; i++) h = ((h << 5) - h + text.charCodeAt(i)) | 0;
    return (h >>> 0).toString(36);
  }

  function wordCount(text) {
    return text.trim().split(/\s+/).filter(Boolean).length;
  }

  function authorKey(container) {
    const el =
      container.querySelector(".update-components-actor__title") ||
      container.querySelector('a[href*="/in/"], a[href*="/company/"]');
    return (el?.textContent || "").replace(/\s+/g, " ").trim();
  }

  function postText(container) {
    const el = container.querySelector(POST_TEXT);
    if (!el) return "";
    return (el.textContent || "").replace(/\s+/g, " ").trim();
  }

  function findTarget(container) {
    for (const selector of INSERTION_TARGETS) {
      const el = container.querySelector(selector);
      if (el) return { parent: el, ref: null };
    }
    const textEl = container.querySelector(POST_TEXT);
    if (textEl?.parentElement) return { parent: textEl.parentElement, ref: textEl };
    return null;
  }

  function decisiveChoice(entry) {
    if (!entry || entry.error || entry.phase || !entry.choice) return null;
    const probabilities = entry.probabilities || {};
    const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const top =
      typeof probabilities[entry.choice] === "number" ? probabilities[entry.choice] : null;
    const runner = entries.find(([name]) => name !== entry.choice)?.[1] ?? 0;
    return top != null && top >= VERDICT_TOP_MIN && top - runner >= VERDICT_GAP_MIN
      ? entry.choice
      : null;
  }

  function syncDim(container, key) {
    const entry = key ? cache.get(key) : null;
    const dim = dimEnabled && DIM_CHOICES.has(decisiveChoice(entry));
    if (!dim) {
      container.removeAttribute("data-clefsift-dim");
      return;
    }
    if (container.parentElement?.closest("[data-clefsift-dim]")) return;
    if (container.querySelector("[data-clefsift-dim]")) return;
    container.setAttribute("data-clefsift-dim", "");
  }

  function renderBadge(container, entry, key) {
    if (container.querySelector("[data-clefsift-badge]")) return;
    const target = findTarget(container);
    if (!target) return;
    const badge = document.createElement("span");
    badge.setAttribute("data-clefsift-badge", key);
    badge.setAttribute("role", "status");
    updateBadge(badge, entry);
    if (target.ref) target.parent.insertBefore(badge, target.ref);
    else target.parent.appendChild(badge);
  }

  function updateBadges(key, entry) {
    for (const badge of document.querySelectorAll(`[data-clefsift-badge="${key}"]`)) {
      updateBadge(badge, entry);
    }
  }

  function updateBadge(badge, entry) {
    const phase = entry.phase || (entry.error ? "error" : "result");
    if (badge.dataset.clefsiftPhase === phase) return;
    badge.dataset.clefsiftPhase = phase;
    badge.setAttribute("aria-busy", String(phase === "queued" || phase === "checking"));
    badge.style.cssText =
      "display:inline-flex;align-items:center;gap:4px;margin-left:6px;padding:1px 8px;border-radius:999px;font:700 10px/16px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#e5e7eb;color-scheme:dark;vertical-align:middle;white-space:nowrap;cursor:default;letter-spacing:.04em";
    if (entry.phase === "short") {
      badge.style.background = "#343c44";
      badge.style.color = "#e5e7eb";
      badge.textContent = "50+ words needed";
      badge.title = "Posts need at least 50 words to be checked. Expand the post to include more text.";
    } else if (entry.phase) {
      const checking = entry.phase === "checking";
      badge.style.background = checking ? "#312e56" : "#343c44";
      badge.style.color = checking ? "#c7d2fe" : "#e5e7eb";
      badge.textContent = checking ? "Checking…" : "Queued";
      badge.title = checking ? "Waiting for the text check to finish." : "Waiting for earlier posts to finish checking.";
    } else if (entry.error) {
      badge.style.background = "#3b2029";
      badge.style.color = "#fecdd3";
      badge.textContent = "N/A";
      badge.title = `Check failed: ${entry.error}`;
    } else {
      const choice = entry.choice;
      const probabilities = entry.probabilities || {};
      const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
      const decisive = decisiveChoice(entry) != null;
      const breakdown = entries
        .map(([name, value]) => `${name}: ${Math.round((value || 0) * 100)}%`)
        .join(", ");
      if (decisive) {
        badge.style.background = BACKGROUNDS[choice] || "#343c44";
        badge.style.color = COLORS[choice] || "#e5e7eb";
        badge.textContent = LABELS[choice] || choice.toUpperCase();
      } else {
        badge.style.background = "#343c44";
        badge.textContent = "UNCERTAIN";
      }
      badge.title = `${decisive ? "" : "Weak margin. "}${breakdown} | leading: ${choice}${entry.model ? ` | model: ${entry.model}` : ""}`;
    }
  }

  function processContainers() {
    if (!enabled) return;
    for (const container of document.querySelectorAll(POST_CONTAINERS)) {
      const badge = container.querySelector("[data-clefsift-badge]");
      if (badge) {
        if (badge.dataset.clefsiftPhase === "short" && wordCount(postText(container)) < FEED_MIN_WORDS) {
          syncDim(container, null);
          continue;
        }
        const key = badge.getAttribute("data-clefsift-badge");
        const cached = cache.get(key);
        if (cached || pending.has(key)) {
          updateBadge(badge, cached || { phase: key === activeKey ? "checking" : "queued" });
          syncDim(container, key);
          continue;
        }
        badge.remove();
      }
      const text = postText(container);
      if (!text) continue;
      const key = hash(`${authorKey(container)}|${text}`);
      if (wordCount(text) < FEED_MIN_WORDS) {
        renderBadge(container, { phase: "short" }, key);
        syncDim(container, key);
        continue;
      }
      const cached = cache.get(key);
      if (cached) {
        renderBadge(container, cached, key);
        syncDim(container, key);
        continue;
      }
      if (!pending.has(key)) {
        if (queue.length >= MAX_QUEUE) continue;
        pending.add(key);
        queue.push({ key, text });
      }
      renderBadge(container, { phase: key === activeKey ? "checking" : "queued" }, key);
      syncDim(container, key);
    }
    pump();
  }

  function withTimeout(promise, ms) {
    return Promise.race([
      promise,
      new Promise((resolve) => setTimeout(() => resolve({ timeout: true }), ms)),
    ]);
  }

  async function pump() {
    if (!enabled || activeKey !== null) return;
    const item = queue.shift();
    if (!item) return;
    activeKey = item.key;
    updateBadges(item.key, { phase: "checking" });
    try {
      const response = await withTimeout(
        chrome.runtime.sendMessage({
          type: "SCAN_TEXT",
          text: item.text,
          source: "linkedin-feed",
          pageUrl: location.href,
        }),
        SCAN_TIMEOUT_MS,
      );
      if (response?.timeout) {
        cache.set(item.key, { error: "Timed out waiting for the decision model." });
      } else if (response?.ok) {
        cache.set(item.key, response.result);
      } else {
        cache.set(item.key, { error: response?.error || "Unknown error." });
      }
    } catch (error) {
      cache.set(item.key, { error: error.message });
    } finally {
      pending.delete(item.key);
      if (enabled) updateBadges(item.key, cache.get(item.key));
      activeKey = null;
      processContainers();
    }
  }

  function schedule() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(processContainers, DEBOUNCE_MS);
  }

  function start() {
    if (observer) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      @keyframes clefsift-feed-spin { to { transform: rotate(360deg); } }
      [data-clefsift-badge][data-clefsift-phase="checking"]::before {
        content: "";
        width: 10px;
        height: 10px;
        flex: none;
        box-sizing: border-box;
        border: 1.5px solid #ffffff66;
        border-top-color: currentColor;
        border-radius: 50%;
        animation: clefsift-feed-spin 0.8s linear infinite;
      }
      [data-clefsift-dim] { opacity: 0.5; transition: opacity 150ms ease; }
      [data-clefsift-dim]:hover { opacity: 1; }
      @media (prefers-reduced-motion: reduce) {
        [data-clefsift-badge][data-clefsift-phase="checking"]::before { animation: none; }
      }
    `;
    document.head.appendChild(style);
    observer = new MutationObserver(schedule);
    observer.observe(document.documentElement, { childList: true, subtree: true });
    processContainers();
  }

  function stop() {
    observer?.disconnect();
    observer = null;
    clearTimeout(debounceTimer);
    for (const item of queue) pending.delete(item.key);
    queue.length = 0;
    document.querySelectorAll("[data-clefsift-badge]").forEach((badge) => badge.remove());
    document.querySelectorAll("[data-clefsift-dim]").forEach((el) => el.removeAttribute("data-clefsift-dim"));
    document.getElementById(STYLE_ID)?.remove();
  }

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !changes[SETTINGS_KEY]) return;
    const settings = changes[SETTINGS_KEY].newValue || {};
    const nowEnabled = !!settings.linkedinFeed && !!settings.model;
    const nowDim = !!settings.dimAiPosts;
    const dimChanged = nowDim !== dimEnabled;
    dimEnabled = nowDim;
    if (nowEnabled === enabled) {
      if (enabled && dimChanged) processContainers();
      return;
    }
    enabled = nowEnabled;
    if (enabled) start();
    else stop();
  });

  chrome.storage.local.get(SETTINGS_KEY).then((stored) => {
    const settings = stored[SETTINGS_KEY] || {};
    dimEnabled = !!settings.dimAiPosts;
    enabled = !!settings.linkedinFeed && !!settings.model;
    if (enabled) start();
  });
})();
