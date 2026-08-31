from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from pi_harness import AgentCallbacks, AgentError, RunResult, Usage
from pi_harness.cli import main


class SinglePromptCLITests(unittest.TestCase):
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
