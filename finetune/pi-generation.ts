import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("before_provider_request", (event, ctx) => {
    const model = ctx.model;
    if (!model || !event.payload || typeof event.payload !== "object" || Array.isArray(event.payload)) {
      throw new Error("Invalid generation request context");
    }
    const payload = event.payload as Record<string, unknown>;
    const prompt = readFileSync(join(ctx.cwd, "prompt.txt"), "utf8").split("Source JSON data:\n", 1)[0];
    const range = prompt.match(/Use (\d+)-(\d+) whitespace-separated words\./);
    if (!range) throw new Error("Missing generation word range");
    const minimum = Number(range[1]);
    const maximum = Number(range[2]);
    if (minimum < 1 || maximum < minimum || maximum > 1000) throw new Error("Invalid generation word range");
    const schema = {
      type: "object",
      properties: { text: {
        type: "string",
        description: `Write ${minimum}-${maximum} whitespace-separated words, aiming for ${Math.round((minimum + maximum) / 2)}. Follow the editing or writing task exactly.`,
        pattern: `^\\S+(?:\\s+\\S+){${minimum - 1},${maximum - 1}}$`,
      } },
      required: ["text"],
      additionalProperties: false,
    };
    let options: Record<string, unknown>;
    if (model.provider === "local" && model.id === "qwen3.8-27b") {
      const localSchema = { ...schema, properties: { text: { type: "string", description: schema.properties.text.description } } };
      options = {
        chat_template_kwargs: { ...(payload.chat_template_kwargs as object ?? {}), enable_thinking: true },
        response_format: { type: "json_schema", json_schema: { name: "clefsift_text", strict: true, schema: localSchema } },
      };
    } else if (model.provider === "openai-codex" && model.id === "gpt-6-luna") {
      options = {
        text: { ...(payload.text as object ?? {}), format: { type: "json_schema", name: "clefsift_text", strict: true, schema } },
      };
    } else {
      throw new Error("Unexpected generation route");
    }
    writeFileSync(join(ctx.cwd, "request-options.json"), JSON.stringify({
      route: `${model.provider}/${model.id}`,
      options,
      policy: "Request-only schema restates the existing JSON and word-range requirements. Local reasoning is enabled for constraint adherence; its server cannot compile the word-range regex. Prompts, responses, acceptance checks, credentials, and global configuration are unchanged",
    }, null, 2) + "\n", { flag: "wx", mode: 0o600 });
    return { ...payload, ...options };
  });
}
