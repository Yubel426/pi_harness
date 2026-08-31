from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from pi_harness import AgentCallbacks, AgentError, RunResult, Usage
from pi_harness.cli import main


class SinglePromptCLITests(unittest.TestCase):
    def test_management_commands_do_not_create_an_agent(self) -> None:
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.get_providers.return_value = []
        with (
            patch("pi_harness.cli.PiClient", return_value=client),
            patch("pi_harness.cli._create_agent") as create,
        ):
            self.assertEqual(main(["providers"]), 0)
        create.assert_not_called()
        client.get_providers.assert_called_once_with()

    def test_logout_requires_an_explicit_provider(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["logout"])

    def test_login_can_use_key_from_stdin_without_echo(self) -> None:
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        output = io.StringIO()
        with (
            patch("pi_harness.cli.PiClient", return_value=client),
            patch("sys.stdin", io.StringIO("test-secret\n")),
            redirect_stdout(output),
        ):
            self.assertEqual(main(["login", "anthropic", "--api-key-stdin"]), 0)
        client.set_api_key.assert_called_once_with("anthropic", "test-secret")
        self.assertNotIn("test-secret", output.getvalue())

    def test_missing_or_blank_prompt_exits_before_client_creation(self) -> None:
        for arguments in ([], ["-p", ""], ["-p", " \n\t"]):
            with (
                self.subTest(arguments=arguments),
                patch("pi_harness.cli._create_agent") as create,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(arguments)
            self.assertEqual(raised.exception.code, 2)
            create.assert_not_called()

    def test_one_prompt_runs_once_then_closes_client(self) -> None:
        agent = Mock()
        agent.run.return_value = RunResult(
            "done", Usage(), "test-model", "resp_1", 1, 0
        )
        with (
            patch("pi_harness.cli._create_agent", return_value=agent),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = main(["-p", "inspect a scene"])
        self.assertEqual(exit_code, 0)
        agent.run.assert_called_once()
        prompt, callbacks = agent.run.call_args.args
        self.assertEqual(prompt, "inspect a scene")
        self.assertIsInstance(callbacks, AgentCallbacks)
        agent.client.close.assert_called_once_with()

    def test_failure_or_interrupt_exits_without_reprompting(self) -> None:
        for failure in (AgentError("test failure"), KeyboardInterrupt()):
            agent = Mock()
            agent.run.side_effect = failure
            with (
                self.subTest(failure=type(failure).__name__),
                patch("pi_harness.cli._create_agent", return_value=agent),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                exit_code = main(["-p", "inspect a scene"])
            self.assertEqual(exit_code, 1)
            agent.run.assert_called_once()
            agent.client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
