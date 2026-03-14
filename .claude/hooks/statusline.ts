/**
 * statusline.ts — ステータスライン + コンテキスト残量の永続化
 *
 * Claude Code の statusLine から stdin で受け取った context_window 情報を
 * /tmp/context_usage.json に書き出し、interoception.sh から読めるようにする。
 * 同時にステータスラインのテキストを stdout に返す。
 */

const CONTEXT_USAGE_PATH = "/tmp/context_usage.json";

interface StatusLineInput {
  session_id?: string;
  model?: { id: string; display_name: string };
  cost?: { total_cost_usd: number };
  context_window?: {
    used_percentage: number;
    remaining_percentage: number;
    context_window_size: number;
    total_input_tokens?: number;
    total_output_tokens?: number;
    current_usage?: {
      input_tokens: number;
      output_tokens: number;
      cache_creation_input_tokens: number;
      cache_read_input_tokens: number;
    };
  };
}

// stdin を読む
const chunks: Buffer[] = [];
for await (const chunk of Bun.stdin.stream()) {
  chunks.push(Buffer.from(chunk));
}
const raw = Buffer.concat(chunks).toString("utf-8").trim();

let data: StatusLineInput;
try {
  data = JSON.parse(raw);
} catch {
  console.log("⚠️ statusline: invalid input");
  process.exit(0);
}

// --- /tmp/context_usage.json に書き出し ---
const ctx = data.context_window;
if (ctx) {
  const usage = {
    used_percentage: ctx.used_percentage ?? 0,
    remaining_percentage: ctx.remaining_percentage ?? 100,
    context_window_size: ctx.context_window_size ?? 200000,
    input_tokens: ctx.current_usage?.input_tokens ?? 0,
    ts: new Date().toISOString(),
    session_id: data.session_id ?? "unknown",
  };
  // 非同期で書く（ステータスライン表示をブロックしない）
  Bun.write(CONTEXT_USAGE_PATH, JSON.stringify(usage)).catch(() => {});
}

// --- ステータスライン表示 ---
const model = data.model?.display_name ?? "?";
const cost = data.cost?.total_cost_usd ?? 0;
const usedPct = ctx?.used_percentage ?? 0;
const tokens = ctx?.current_usage?.input_tokens ?? 0;

// コンテキストバーの色付け
const contextLabel = usedPct >= 80 ? "🔴" : usedPct >= 60 ? "🟡" : "🟢";

const parts = [
  `🤖 ${model}`,
  `💰 $${cost.toFixed(2)}`,
  `${contextLabel} ${tokens.toLocaleString()}tok (${usedPct}%)`,
];

console.log(parts.join(" | "));
