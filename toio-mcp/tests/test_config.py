from toio_mcp.config import ToioConfig


def test_from_env_defaults(monkeypatch):
    monkeypatch.delenv("TOIO_CUBE_NAME", raising=False)
    monkeypatch.delenv("TOIO_DRY_RUN", raising=False)
    config = ToioConfig.from_env()
    assert config.cube_name is None
    assert config.scan_timeout == 5
    assert config.max_speed == 70
    assert config.dry_run is False


def test_from_env_custom(monkeypatch):
    monkeypatch.setenv("TOIO_CUBE_NAME", "123")
    monkeypatch.setenv("TOIO_SCAN_TIMEOUT", "7")
    monkeypatch.setenv("TOIO_MAX_SPEED", "55")
    monkeypatch.setenv("TOIO_DRY_RUN", "1")
    config = ToioConfig.from_env()
    assert config.cube_name == "123"
    assert config.scan_timeout == 7
    assert config.max_speed == 55
    assert config.dry_run is True
