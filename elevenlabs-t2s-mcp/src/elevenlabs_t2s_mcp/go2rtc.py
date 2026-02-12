"""Auto-download and manage go2rtc for audio backchannel."""

import asyncio
import json
import logging
import platform
import stat
import subprocess
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = "https://api.github.com/repos/AlexxIT/go2rtc/releases/latest"

PLATFORM_MAP = {
    ("linux", "x86_64"): "go2rtc_linux_amd64",
    ("linux", "aarch64"): "go2rtc_linux_arm64",
    ("linux", "armv7l"): "go2rtc_linux_arm",
    ("linux", "armv6l"): "go2rtc_linux_armv6",
    ("darwin", "x86_64"): "go2rtc_mac_amd64",
    ("darwin", "arm64"): "go2rtc_mac_arm64",
    ("windows", "amd64"): "go2rtc_win64",
    ("windows", "x86"): "go2rtc_win32",
}


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "embodied-claude" / "go2rtc"


def default_bin_path() -> Path:
    name = "go2rtc.exe" if platform.system().lower() == "windows" else "go2rtc"
    return default_cache_dir() / name


def default_config_path() -> Path:
    return default_cache_dir() / "go2rtc.yaml"


def detect_platform() -> str:
    """Detect the go2rtc asset name for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    key = (system, machine)
    asset_name = PLATFORM_MAP.get(key)
    if not asset_name:
        raise RuntimeError(f"Unsupported platform: {system}/{machine}")
    return asset_name


def _get_download_url(asset_name: str) -> str:
    """Get download URL from GitHub Releases API."""
    req = urllib.request.Request(
        GITHUB_RELEASES_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        release = json.loads(resp.read())

    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name == asset_name or name.startswith(asset_name):
            return asset["browser_download_url"]

    raise RuntimeError(
        f"Asset '{asset_name}' not found in latest release"
    )


def _download_file(url: str, dest: Path) -> None:
    """Download a file from URL to dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.rename(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_binary(bin_path: Path | None = None) -> Path:
    """Ensure go2rtc binary exists, downloading if needed."""
    if bin_path is None:
        bin_path = default_bin_path()

    if bin_path.exists():
        return bin_path

    logger.info("go2rtc binary not found, downloading...")
    asset_name = detect_platform()
    url = _get_download_url(asset_name)
    logger.info("Downloading go2rtc from %s", url)

    is_zip = url.endswith(".zip")
    if is_zip:
        zip_path = bin_path.with_suffix(".zip")
        _download_file(url, zip_path)
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            # Find the binary inside the zip
            for name in zf.namelist():
                if "go2rtc" in name.lower():
                    with zf.open(name) as src, open(bin_path, "wb") as dst:
                        dst.write(src.read())
                    break
            else:
                raise RuntimeError("go2rtc binary not found in zip")
        zip_path.unlink()
    else:
        _download_file(url, bin_path)

    # Make executable on Unix
    if platform.system().lower() != "windows":
        bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    logger.info("go2rtc downloaded to %s", bin_path)
    return bin_path


def generate_config(
    config_path: Path,
    stream_name: str,
    camera_host: str,
    username: str,
    password: str,
    ffmpeg_bin: str | None = None,
) -> Path:
    """Generate go2rtc.yaml config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_ffmpeg = ffmpeg_bin or "ffmpeg"
    content = (
        f"streams:\n"
        f"  {stream_name}:\n"
        f"    - rtsp://{username}:{password}@{camera_host}:554/stream1\n"
        f"    - tapo://{password}@{camera_host}\n"
        f"\n"
        f"ffmpeg:\n"
        f"  bin: {resolved_ffmpeg}\n"
        f"\n"
        f"api:\n"
        f"  listen: \":1984\"\n"
        f"\n"
        f"log:\n"
        f"  level: info\n"
    )
    config_path.write_text(content)
    logger.info("go2rtc config written to %s", config_path)
    return config_path


class Go2RTCProcess:
    """Manage go2rtc daemon lifecycle."""

    def __init__(self, bin_path: Path, config_path: Path, api_url: str = "http://localhost:1984"):
        self._bin_path = bin_path
        self._config_path = config_path
        self._api_url = api_url
        self._process: subprocess.Popen | None = None

    def is_running(self) -> bool:
        """Check if go2rtc is already responding."""
        try:
            req = urllib.request.Request(f"{self._api_url}/api", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    async def start(self) -> None:
        """Start go2rtc as a background process."""
        if self.is_running():
            logger.info("go2rtc already running at %s", self._api_url)
            return

        self._process = subprocess.Popen(
            [str(self._bin_path), "-config", str(self._config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Wait briefly and verify it started
        await asyncio.sleep(1.5)

        if self._process.poll() is not None:
            stderr = self._process.stderr.read().decode() if self._process.stderr else ""
            raise RuntimeError(f"go2rtc exited immediately: {stderr}")

        if not self.is_running():
            stderr = ""
            if self._process.stderr:
                self._process.stderr.close()
            self._process.terminate()
            raise RuntimeError("go2rtc started but not responding")

        logger.info("go2rtc started (pid=%d)", self._process.pid)

    def stop(self) -> None:
        """Stop go2rtc process."""
        if self._process is None or self._process.poll() is not None:
            return
        logger.info("Stopping go2rtc (pid=%d)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)
