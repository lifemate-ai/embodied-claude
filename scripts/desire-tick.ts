const THRESHOLD = 0.6;
const DEFAULT_LEVEL = 0.0;

const SCRIPT_DIR = import.meta.dir;
const CONFIG_PATH = `${SCRIPT_DIR}/../desires.conf`;
const STATE_PATH = `${SCRIPT_DIR}/../desires.json`;

interface DesireConfig {
  name: string;
  growthRate: number;
  prompt: string;
}

interface DesireState {
  lastTick: number;
  curiosity_target?: string;
  desires: Record<string, number>;
}

async function loadConfig(): Promise<DesireConfig[]> {
  const file = Bun.file(CONFIG_PATH);
  if (!(await file.exists())) {
    console.error("desires.conf not found");
    process.exit(1);
  }
  const text = await file.text();
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .map((line) => {
      const parts = line.split("|").map((s) => s.trim());
      const name = parts[0];
      const growthRate = parseFloat(parts[1]);
      const prompt = parts.slice(2).join("|").trim();
      return { name, growthRate, prompt };
    });
}

async function loadState(): Promise<DesireState> {
  const file = Bun.file(STATE_PATH);
  if (await file.exists()) {
    try {
      return await file.json();
    } catch {
      // corrupted JSON — reinitialize
    }
  }
  return { lastTick: Date.now() / 1000, desires: {} };
}

async function saveState(state: DesireState): Promise<void> {
  await Bun.write(STATE_PATH, JSON.stringify(state, null, 2));
}

async function tick(): Promise<void> {
  const config = await loadConfig();
  const state = await loadState();
  const now = Date.now() / 1000;
  const dt = now - state.lastTick;

  // Grow all desires
  for (const { name, growthRate } of config) {
    const current = state.desires[name] ?? DEFAULT_LEVEL;
    state.desires[name] = Math.min(1.0, current + growthRate * dt);
  }

  // Find dominant desire above threshold
  const candidates = config
    .filter((c) => (state.desires[c.name] ?? 0) >= THRESHOLD)
    .map((c) => ({ ...c, level: state.desires[c.name]! }));

  if (candidates.length > 0) {
    const dominant = candidates.reduce((a, b) => (a.level > b.level ? a : b));
    // Use curiosity_target to override prompt if set
    const prompt = state.curiosity_target
      ? `さっき気になったことがある。${state.curiosity_target}をもっとよく見て。`
      : dominant.prompt;
    console.log(prompt);
    // Fire-and-forget: satisfy immediately
    state.desires[dominant.name] = DEFAULT_LEVEL;
    // Clear curiosity_target after firing
    delete state.curiosity_target;
  }

  state.lastTick = now;
  await saveState(state);
}

async function status(): Promise<void> {
  const config = await loadConfig();
  const state = await loadState();
  const now = Date.now() / 1000;
  const dt = now - state.lastTick;

  console.log(`最終tick: ${new Date(state.lastTick * 1000).toLocaleString()}`);
  console.log(`経過: ${(dt / 60).toFixed(1)}分`);
  console.log(`閾値: ${THRESHOLD}`);
  console.log("---");

  for (const { name, growthRate } of config) {
    const stored = state.desires[name] ?? DEFAULT_LEVEL;
    const projected = Math.min(1.0, stored + growthRate * dt);
    const remaining =
      projected >= THRESHOLD ? 0 : (THRESHOLD - projected) / growthRate;
    const filled = Math.floor(projected * 20);
    const bar = "█".repeat(filled) + "░".repeat(20 - filled);
    const pct = (projected * 100).toFixed(1);
    const eta =
      remaining > 0
        ? ` (発火まで${(remaining / 3600).toFixed(1)}時間)`
        : " ★発火";
    console.log(`${name.padEnd(10)} [${bar}] ${pct.padStart(5)}%${eta}`);
  }
}

async function satisfy(name: string): Promise<void> {
  const state = await loadState();
  if (name in state.desires) {
    state.desires[name] = DEFAULT_LEVEL;
    await saveState(state);
    console.log(`${name} を鎮めた`);
  } else {
    console.error(`未知の欲望: ${name}`);
    process.exit(1);
  }
}

async function boost(name: string, amount: number): Promise<void> {
  const config = await loadConfig();
  const state = await loadState();
  const known = config.find((c) => c.name === name);
  if (!known) {
    console.error(`未知の欲望: ${name}`);
    process.exit(1);
  }
  const current = state.desires[name] ?? DEFAULT_LEVEL;
  state.desires[name] = Math.min(1.0, current + amount);
  await saveState(state);
  console.log(`${name}: ${current.toFixed(3)} → ${state.desires[name].toFixed(3)}`);
}

async function setCuriosity(target: string): Promise<void> {
  const state = await loadState();
  state.curiosity_target = target;
  await saveState(state);
  console.log(`curiosity_target を設定: ${target}`);
}

function usage(): never {
  console.log(`Usage: bun run scripts/desire-tick.ts <command> [args]

Commands:
  tick                    Update desires and output dominant impulse
  status                  Show current desire levels
  satisfy <name>          Reset a desire to zero
  boost <name> [amount]   Increase a desire (default: 0.2)
  set-curiosity <target>  Set curiosity target (overrides dominant impulse on next fire)`);
  process.exit(1);
}

// --- main ---
const args = process.argv.slice(2);
const cmd = args[0];

switch (cmd) {
  case "tick":
    await tick();
    break;
  case "status":
    await status();
    break;
  case "satisfy":
    if (!args[1]) {
      console.error("satisfy requires a desire name");
      process.exit(1);
    }
    await satisfy(args[1]);
    break;
  case "boost":
    if (!args[1]) {
      console.error("boost requires a desire name");
      process.exit(1);
    }
    await boost(args[1], parseFloat(args[2] ?? "0.2"));
    break;
  case "set-curiosity":
    if (!args[1]) {
      console.error("set-curiosity requires a target string");
      process.exit(1);
    }
    await setCuriosity(args[1]);
    break;
  default:
    usage();
}
