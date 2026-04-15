#!/bin/bash
# garmin-hr-cache.sh — Fetch latest heart rate from Garmin and cache it
# Run via cron every 5 minutes: */5 * * * * /path/to/garmin-hr-cache.sh
# Writes latest HR to /tmp/garmin_hr_latest.txt for interoception.sh to read

CACHE_FILE="/tmp/garmin_hr_latest.txt"
GARMIN_MCP_DIR="/Users/mizushima/repo/garmin-health-mcp"
export PATH="/opt/homebrew/bin:$PATH"

cd "$GARMIN_MCP_DIR" || exit 1

# Load credentials
if [ -f "$GARMIN_MCP_DIR/.env" ]; then
    export $(grep -v '^#' "$GARMIN_MCP_DIR/.env" | xargs)
fi

# Call garmin-connect to get today's heart rate and extract latest value
node -e "
const GarminConnect = require('garmin-connect');
const { GarminConnect: GC } = GarminConnect;
const { readFileSync, existsSync } = require('fs');
const { join } = require('path');
const { homedir } = require('os');

(async () => {
  const TOKEN_PATH = join(homedir(), '.garmin-token.json');
  const client = new GC({
    username: process.env.GARMIN_EMAIL || '',
    password: process.env.GARMIN_PASSWORD || '',
  });

  await client.login();

  const hr = await client.getHeartRate(new Date());
  const resting = hr?.restingHeartRate;
  const values = hr?.heartRateValues || [];
  let latest = null;
  let latestTs = null;
  for (const entry of values) {
    if (entry && entry[1] > 0) { latest = entry[1]; latestTs = entry[0]; }
  }
  let out = '';
  if (latest) {
    const when = latestTs ? new Date(latestTs).toTimeString().slice(0,5) : '';
    out = latest + 'bpm' + (when ? '@' + when : '');
  } else if (resting) {
    out = resting + 'bpm(resting)';
  }
  if (out) console.log(out);
})().catch(() => {});
" 2>/dev/null | grep -oE '[0-9]+bpm(@[0-9:]+|\(resting\))?' | tail -1 > "$CACHE_FILE"
