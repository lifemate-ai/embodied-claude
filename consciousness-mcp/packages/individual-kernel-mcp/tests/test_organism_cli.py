"""The clock has to be runnable without a conversation.

The organism daemon is the only producer of an autonomous tick, and its whole
point is to move when nobody is talking to the agent. Until now it was reachable
only through the `organism_step` MCP tool, which needs a language model to call
it -- so the non-verbal clock could only be wound by the very thing it was
supposed to be independent of. A scheduler needs a command it can run.
"""

from __future__ import annotations

from social_core.db import SocialDB

from individual_kernel_mcp import hook_cli
from individual_kernel_mcp.tick import TickProducer


class TestTheCommandExists:
    def test_the_cli_accepts_it(self) -> None:
        # cron invokes `efpf-hook organism-step`; if the command is not in the
        # accepted set the entry point rejects it before anything runs.
        assert "organism-step" in hook_cli.COMMANDS


class TestTheFlagGovernsIt:
    def test_it_declines_while_the_flag_is_off(
        self, social_db: SocialDB, monkeypatch
    ) -> None:
        monkeypatch.setattr(hook_cli, "organism_enabled", lambda: False)

        result = hook_cli.organism_step(social_db, TickProducer(social_db))

        assert result["ran"] is False
        assert "organism_daemon" in result["reason"]

    def test_it_turns_the_clock_while_the_flag_is_on(
        self, social_db: SocialDB, monkeypatch
    ) -> None:
        monkeypatch.setattr(hook_cli, "organism_enabled", lambda: True)

        result = hook_cli.organism_step(social_db, TickProducer(social_db))

        # The daemon decides for itself whether to ignite; what matters here is
        # that the turn happened and reported its reasoning.
        assert result["ran"] is True
        assert "ignited" in result


class TestItIsSafeToRunFromCron:
    def test_a_dry_run_changes_nothing(self, social_db: SocialDB, monkeypatch) -> None:
        monkeypatch.setattr(hook_cli, "organism_enabled", lambda: True)
        before = social_db.fetchone("SELECT COUNT(*) AS n FROM organism_runs")["n"]

        hook_cli.organism_step(social_db, TickProducer(social_db), dry_run=True)

        after = social_db.fetchone("SELECT COUNT(*) AS n FROM organism_runs")["n"]
        assert after == before
