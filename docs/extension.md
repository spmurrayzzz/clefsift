# Extension

The Chrome Manifest V3 extension needs no build step.
[Installation](../README.md#use-the-extension) requires a SystemOne-compatible server.

## Scans and settings

- Right-click selected text to check the selection.
- Right-click a page to check its body text.
- Open the popup to paste and check text.
- Enable LinkedIn scanning in settings to add feed badges.
- Enable post dimming to dim decisive AI and AI-assisted results.

Manual scans require ten words. LinkedIn posts require 50 extracted words.
Requests contain at most 50,000 characters. The server token limit can be lower.

Model discovery accepts `models[].name` or `data[].id` from `GET {base}/models`.
The picker saves the exact model ID.
Use **Custom model…** when the server does not provide a model list.
A stale model warning does not save a replacement automatically.

**Test connection** uses saved settings. Save connection changes before the test.
Enable LinkedIn scanning only when you want automatic requests.
Reload LinkedIn after model or settings changes to clear cached results and errors.

Badges show `Queued`, then `Checking…`, then a result.
Short posts show `50+ words needed` without an API request.
The feed waits up to 120 seconds. A timeout does not cancel the worker request.

Dimming applies to whole posts with decisive AI or AI-assisted results.
Point at a dimmed post to restore its opacity.
Mixed, human, uncertain, and failed results do not dim posts.
The worker restores enabled scanning after an extension reload when origin access remains available.
LinkedIn DOM changes can break post discovery.

A new unpacked installation has separate settings and permissions.
Loading this repo does not migrate another installation automatically.

## Privacy and permissions

The extension sends scan text, a model ID, and the classification question to your configured server.
It does not send page URLs or author names as separate request fields.
Full-page scans can include sensitive page text. Do not scan text that you cannot send to that server.
Hosted-server retention depends on its provider.
Use HTTPS for remote servers. HTTP does not encrypt traffic.

Settings and API keys use `chrome.storage.local`.
The extension does not encrypt stored keys at the application level.
Local history retains the latest 20 manual and connection-test results.
Each entry contains a 140-character snippet, timestamp, result, model ID, and URL when available.
The popup shows eight history entries. Feed scans do not enter history.

The extension has no telemetry and does not read the clipboard.

| Permission | Purpose |
|---|---|
| `contextMenus` | Selection and page scan menus |
| `storage` | Local settings and history |
| `activeTab` | User-requested scans on the active page |
| `scripting` | Scan results and the optional feed script |
| Optional HTTP/HTTPS origins | Access to the selected server and LinkedIn, after approval |

No host access is granted at installation.
Connection, save, and toggle actions request access to the required origins.

## Decision API

The base URL normally ends in `/v1`.
Scans use `POST {base}/systemone` with an `Idempotency-Key` UUID.
A saved API key adds an `Authorization: Bearer` header.

The request contains `model`, `state`, and `questions.classification`.
The classification question is the complete choice object in `extension/background.js` and `finetune/common.py`.
Keep that object unchanged for comparable results.
The server response must contain:

```json
{
  "answers": {
    "classification": {
      "type": "choice",
      "choice": "human",
      "probabilities": {
        "human": 0.7,
        "ai-assisted": 0.1,
        "mixed": 0.1,
        "ai": 0.1
      }
    }
  }
}
```

The four labels describe document-level classes.
The model question asks about writing style, not topic or quality.

A result is decisive when its top probability is at least `0.35` and its margin is at least `0.10`.
Otherwise, the main result shows `UNCERTAIN`.
History chips show the raw choice without this rule.
Probabilities and verdicts do not establish authorship.
