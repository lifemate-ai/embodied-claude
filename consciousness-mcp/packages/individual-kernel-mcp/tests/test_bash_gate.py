"""The bash gate has to read the command, not a string that resembles one.

The gate decides whether a shell command is inspection or a side effect, and it
fails closed. That makes a false positive expensive in a specific way: an
ordinary read is refused, and the refusal looks like a policy decision rather
than a parsing accident. Every case here is a command that was actually issued.
"""

from __future__ import annotations

from individual_kernel_mcp.agency import _bash_is_read_only, is_external_tool


class TestQuotedOperators:
    """An operator inside quotes is an argument, not shell syntax."""

    def test_a_pipe_inside_a_quoted_regex_is_not_a_pipeline(self) -> None:
        # grep alternation. Splitting the raw string cut the command in half and
        # left an unbalanced quote, which the gate then read as a write.
        assert _bash_is_read_only('grep -n "alpha\\|beta" file.txt')

    def test_a_semicolon_inside_a_quoted_argument_is_not_a_separator(self) -> None:
        assert _bash_is_read_only('grep -n "alpha;beta" file.txt')

    def test_an_arrow_inside_quotes_is_not_a_redirection(self) -> None:
        assert _bash_is_read_only('grep -n "alpha > beta" file.txt')

    def test_an_ampersand_inside_quotes_is_not_a_background_job(self) -> None:
        assert _bash_is_read_only('grep -n "alpha && beta" file.txt')


class TestRealOperatorsStillCount:
    """Quote awareness must not become permissiveness."""

    def test_every_stage_of_a_real_pipeline_is_checked(self) -> None:
        assert _bash_is_read_only("cat file.txt | wc -l")
        assert not _bash_is_read_only("cat file.txt | tee copy.txt")

    def test_a_real_redirection_is_a_write(self) -> None:
        assert not _bash_is_read_only("echo hi > out.txt")
        assert not _bash_is_read_only("echo hi >> out.txt")

    def test_a_writing_command_after_a_separator_is_a_write(self) -> None:
        assert not _bash_is_read_only("ls /tmp; rm -rf /tmp/x")

    def test_command_substitution_is_still_refused(self) -> None:
        assert not _bash_is_read_only("echo $(rm -rf /tmp/x)")
        assert not _bash_is_read_only("echo `rm -rf /tmp/x`")

    def test_discarding_output_stays_allowed(self) -> None:
        assert _bash_is_read_only("ls /tmp > /dev/null")
        assert _bash_is_read_only("ls /tmp 2>/dev/null")
        assert _bash_is_read_only("ls /tmp > /dev/null 2>&1")


class TestGitGlobalOptions:
    """A global option must not be mistaken for the subcommand."""

    def test_a_path_option_does_not_hide_the_subcommand(self) -> None:
        # `-C` takes a value, so the first non-flag token is the path.
        assert _bash_is_read_only("git -C /tmp/repo status")

    def test_a_config_option_does_not_hide_the_subcommand(self) -> None:
        assert _bash_is_read_only("git -c core.pager=cat log -1")

    def test_an_attached_value_option_does_not_hide_the_subcommand(self) -> None:
        assert _bash_is_read_only("git --git-dir=/tmp/repo/.git rev-parse HEAD")

    def test_a_writing_subcommand_behind_an_option_is_still_refused(self) -> None:
        assert not _bash_is_read_only("git -C /tmp/repo commit -m x")
        assert not _bash_is_read_only("git -C /tmp/repo push")


class TestGateIntegration:
    def test_the_gate_calls_a_quoted_pipe_internal(self) -> None:
        assert not is_external_tool("Bash", {"command": 'grep "alpha\\|beta" x'})

    def test_the_gate_still_calls_a_write_external(self) -> None:
        assert is_external_tool("Bash", {"command": "echo hi > out.txt"})
