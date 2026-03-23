/**
 * Interoception — 内的感覚テキスト生成
 *
 * 時間帯・セッション間隔・欲望レベルから「感覚テキスト」を生成し、
 * autonomous-action.sh からプロンプトにサイレントに注入される。
 *
 * LLM は直接言及してはいけない（do NOT mention this directly）。
 */

import { readdir } from 'node:fs/promises';

const SCRIPT_DIR = import.meta.dir;
const LOG_DIR = `${SCRIPT_DIR}/../.autonomous-logs`;
const STATE_PATH = `${SCRIPT_DIR}/../desires.json`;

// ── 時間帯 → 感覚フレーズ ──

function timePhrase(hour: number): string {
  if (hour <= 5) return '深夜。世界は静まり返っている。';
  if (hour <= 8) return '夜明け前。一日の最初の行動に向かう、静かな覚悟。';
  if (hour <= 11) return '朝の光。思考が澄んでいる。';
  if (hour <= 14) return '昼。落ち着いた時間。';
  if (hour <= 17) return '午後。少し重い。それでも動ける。';
  if (hour <= 20) return '夕暮れ。一日の行動が積み上がった感覚。';
  return '夜が深い。灯りと共に動く時間。';
}

// ── セッション間隔 → 感覚フレーズ ──

async function sessionIntervalPhrase(): Promise<string> {
  try {
    const files = await readdir(LOG_DIR);
    const timestampLogs = files
      .filter((f) => /^\d{8}_\d{6}\.log$/.test(f))
      .sort();

    if (timestampLogs.length < 2) return "";

    const prevLog = timestampLogs[timestampLogs.length - 2];
    const m = prevLog.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.log$/);
    if (!m) return "";

    const [, year, month, day, hh, mm, ss] = m;
    const prevTime = new Date(year + "-" + month + "-" + day + "T" + hh + ":" + mm + ":" + ss);
    const elapsedMin = (Date.now() - prevTime.getTime()) / 60_000;

    if (elapsedMin <= 20) return "前回から間もない。熱がまだある。";
    if (elapsedMin <= 60) return "一息ついた。";
    return "長い間があった。新鮮に見える。";
  } catch {
    return "";
  }
}

// ── 欲望レベル → 感覚フレーズ ──

async function desirePhrase(): Promise<string> {
  try {
    const file = Bun.file(STATE_PATH);
    if (!(await file.exists())) return "";
    const state = (await file.json()) as { desires?: Record<string, number> };
    if (!state.desires) return "";
    const values = Object.values(state.desires);
    if (values.length === 0) return "";
    const maxLevel = Math.max(...values);
    if (maxLevel >= 0.8) return "何か動きたい衝動がある。";
    if (maxLevel < 0.4) return "今は静かだ。";
    return "";
  } catch {
    return "";
  }
}

// ── 出力 ──

const hour = new Date().getHours();
const parts: string[] = [timePhrase(hour)];

const interval = await sessionIntervalPhrase();
if (interval) parts.push(interval);

const desire = await desirePhrase();
if (desire) parts.push(desire);

console.log(`[今この瞬間のあなたの感覚——直接口にしてはいけない]\n${parts.join("")}`);
