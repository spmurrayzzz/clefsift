import json
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from uuid import uuid4

PORT = 8787
API_KEY = "test-key"
MODELS = [
    {"name": "mock-clef", "displayName": "Mock Clef", "description": "Local mock of a Clef decision model"},
    {"name": "mock-jev", "displayName": "Mock Jev", "description": "Local mock of a Jev decision model"},
]
CATEGORIES = ["human", "ai-assisted", "mixed", "ai"]
MODEL_NAMES = {m["name"] for m in MODELS}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Request-Id, X-System-One-Credits, X-Idempotency-Replayed")
        self.send_header("X-Request-Id", str(uuid4()))
        if status == 200:
            self.send_header("X-System-One-Credits", "1")
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {API_KEY}"

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Idempotency-Key")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            if not self.authed():
                self.send_json(401, {"detail": "Invalid API key"})
                return
            print(f"GET  {path} -> 200 models={len(MODELS)}", flush=True)
            self.send_json(200, {"models": MODELS})
            return
        self.send_json(404, {"detail": "Not found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path != "/v1/systemone":
            self.send_json(404, {"detail": "Not found"})
            return
        if not self.authed():
            self.send_json(401, {"detail": "Invalid API key"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"detail": "Invalid JSON"})
            return
        idem = self.headers.get("Idempotency-Key", "-")
        state = request.get("state", "")
        if not isinstance(state, str) or not state.strip():
            self.send_json(422, {"detail": [{"loc": ["body", "state"], "msg": "state must be a non-empty string", "type": "value_error"}]})
            return
        model = request.get("model")
        if model not in MODEL_NAMES:
            self.send_json(422, {"detail": [{"loc": ["body", "model"], "msg": "Unknown model", "type": "value_error"}]})
            return
        questions = request.get("questions", {})
        seed = zlib.crc32(state.encode())
        choice = CATEGORIES[seed % 4]
        top = min(0.97, 0.62 + (seed % 30) / 100)
        rest = round((1 - top) / 3, 3)
        probabilities = {c: (round(top, 3) if c == choice else rest) for c in CATEGORIES}
        remainder = next(c for c in reversed(CATEGORIES) if c != choice)
        probabilities[remainder] = round(1 - sum(p for c, p in probabilities.items() if c != remainder), 3)
        payload = {
            "model": model,
            "answers": {
                qid: {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": round(top - 0.05, 3),
                }
                for qid in questions
            },
            "usage": {"input_tokens": max(1, len(state) // 4), "output_tokens": 0},
        }
        words = len(state.split())
        print(f"POST {path} -> 200 idem={idem} model={model} words={words} choice={choice}", flush=True)
        self.send_json(200, payload)


def main():
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"ClefSift mock listening on http://localhost:{PORT}", flush=True)
    print(f"API key: {API_KEY}", flush=True)
    print(f"Models: {', '.join(sorted(MODEL_NAMES))}", flush=True)
    print("Set the extension base URL to http://localhost:8787/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
