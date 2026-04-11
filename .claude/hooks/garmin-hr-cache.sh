#!/bin/bash
# garmin-hr-cache.sh — Fetch latest heart rate from Garmin and cache it
# Run via cron every 5 minutes: */5 * * * * /path/to/garmin-hr-cache.sh
# Writes latest HR to /tmp/garmin_hr_latest.txt for interoception.sh to read

CACHE_FILE="/tmp/garmin_hr_latest.txt"
GARMIN_MCP_DIR="/home/mizushima/repo/garmin-health-mcp"

cd "$GARMIN_MCP_DIR" || exit 1

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

  if (existsSync(TOKEN_PATH)) {
    const token = JSON.parse(readFileSync(TOKEN_PATH, 'utf-8'));
    await client.loadToken(token);
  }

  const hr = await client.getHeartRate(new Date());
  const values = hr?.heartRateValues || [];
  let latest = null;
  for (const entry of values) {
    if (entry && entry[1] > 0) latest = entry[1];
  }
  if (latest) {
    console.log(latest + 'bpm');
  }
})().catch(() => {});
" 2>/dev/null > "$CACHE_FILE"
