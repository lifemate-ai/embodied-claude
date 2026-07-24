#!/usr/bin/env python3
"""Start one MCP server over stdio and verify its live protocol surface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Name used in the JSON report")
    parser.add_argument("--command", help="Server executable")
    parser.add_argument("--arg", action="append", default=[], help="Server argument")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--config", type=Path, help="Read a server from an MCP JSON config")
    parser.add_argument("--server", help="Server name within --config")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--remember-roundtrip", action="store_true")
    return parser


def _parse_environment(values: Sequence[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for value in values:
        key, separator, raw_value = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid --env value: {value!r}")
        environment[key] = raw_value
    return environment


def _load_config_server(
    path: Path,
    server_name: str,
) -> tuple[str, list[str], dict[str, str]]:
    raw = json.loads(path.read_text())
    servers = raw.get("mcpServers", {})
    server = servers.get(server_name) if isinstance(servers, Mapping) else None
    if not isinstance(server, Mapping):
        raise TypeError(f"server {server_name!r} is not present in {path}")
    command = server.get("command")
    args = server.get("args", [])
    environment = server.get("env", {})
    if not isinstance(command, str) or not isinstance(args, list):
        raise TypeError(f"server {server_name!r} has an invalid command shape")
    if not isinstance(environment, Mapping):
        raise TypeError(f"server {server_name!r} has an invalid env object")
    return (
        command,
        [str(item) for item in args],
        {str(key): str(value) for key, value in environment.items()},
    )


def _isolated_environment(state_dir: Path | None) -> dict[str, str]:
    if state_dir is None:
        return {}
    state_dir.mkdir(parents=True, exist_ok=True)
    return {
        "MEMORY_DB_PATH": str(state_dir / "memory.db"),
        "SOCIAL_DB_PATH": str(state_dir / "social.db"),
        "DESIRES_PATH": str(state_dir / "desires.json"),
        "MEMORY_HTTP_PORT": "0",
    }


def _resolve_server(
    args: argparse.Namespace,
) -> tuple[str, str, list[str], dict[str, str]]:
    if args.config or args.server:
        if not args.config or not args.server:
            raise ValueError("--config and --server must be used together")
        command, command_args, environment = _load_config_server(
            args.config,
            args.server,
        )
        name = args.name or args.server
    else:
        if not args.name or not args.command:
            raise ValueError("--name and --command are required without --config")
        name = args.name
        command = args.command
        command_args = list(args.arg)
        environment = {}
    environment.update(_parse_environment(args.env))
    environment.update(_isolated_environment(args.state_dir))
    return name, command, command_args, environment


async def _probe(
    *,
    command: str,
    args: list[str],
    environment: Mapping[str, str],
    cwd: Path | None,
    remember_roundtrip: bool,
    timeout: float,
) -> tuple[int, bool]:
    child_environment = os.environ.copy()
    child_environment.update(environment)
    parameters = StdioServerParameters(
        command=command,
        args=args,
        env=child_environment,
        cwd=cwd,
    )
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_log:
        async with asyncio.timeout(timeout):
            async with stdio_client(parameters, errlog=error_log) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_count = len(tools.tools)
                    if tool_count == 0:
                        raise RuntimeError("server initialized but exposed no tools")

                    roundtrip_ok = False
                    if remember_roundtrip:
                        tool_names = {tool.name for tool in tools.tools}
                        if "remember" not in tool_names:
                            raise RuntimeError("memory server does not expose remember")
                        result = await session.call_tool(
                            "remember",
                            {
                                "content": "embodied-claude doctor live probe",
                                "emotion": "neutral",
                                "importance": 1,
                                "category": "technical",
                                "auto_link": False,
                            },
                        )
                        if result.isError:
                            raise RuntimeError("memory remember round-trip failed")
                        roundtrip_ok = True
                    return tool_count, roundtrip_ok


def _redact(message: str, environment: Mapping[str, str]) -> str:
    redacted = message
    for value in environment.values():
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def _error_message(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        messages = [_error_message(child) for child in error.exceptions]
        return "; ".join(message for message in messages if message)
    return str(error) or type(error).__name__


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    name = args.name or args.server or "unknown"
    environment: dict[str, str] = {}
    try:
        name, command, command_args, environment = _resolve_server(args)
        tool_count, roundtrip_ok = asyncio.run(
            _probe(
                command=command,
                args=command_args,
                environment=environment,
                cwd=args.cwd,
                remember_roundtrip=args.remember_roundtrip,
                timeout=args.timeout,
            )
        )
        report: dict[str, Any] = {
            "ok": True,
            "server": name,
            "tool_count": tool_count,
            "remember_roundtrip": roundtrip_ok,
        }
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - this is the probe's JSON error boundary
        report = {
            "ok": False,
            "server": name,
            "error": _redact(_error_message(error), environment),
        }
        exit_code = 1
    json.dump(report, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
