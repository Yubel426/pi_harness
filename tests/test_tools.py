from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pi_harness.tools import ToolContext, create_bash_tool


class BashToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cwd = Path.cwd().resolve()

    def test_combines_stdout_stderr_and_reports_nonzero(self) -> None:
        tool = create_bash_tool(self.cwd)
        result = tool.execute(
            {"command": "printf out; printf err >&2; exit 7"}, ToolContext(self.cwd)
        )
        self.assertTrue(result.is_error)
        self.assertIn("out", result.output)
        self.assertIn("err", result.output)
        self.assertIn("code 7", result.output)

    def test_timeout_kills_command(self) -> None:
        tool = create_bash_tool(self.cwd)
        started = time.monotonic()
        result = tool.execute(
            {"command": "sleep 5", "timeout": 0.1}, ToolContext(self.cwd)
        )
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.output)

    def test_truncates_tail_and_keeps_full_output(self) -> None:
        tool = create_bash_tool(self.cwd, max_lines=2, max_bytes=1024)
        result = tool.execute(
            {"command": "printf 'one\\ntwo\\nthree\\n'"}, ToolContext(self.cwd)
        )
        self.assertFalse(result.is_error)
        self.assertNotIn("one", result.output)
        self.assertIn("two", result.output)
        self.assertIn("three", result.output)
        full_output = Path(result.metadata["full_output_path"])
        self.assertEqual(full_output.read_text(), "one\ntwo\nthree\n")
        full_output.unlink()

    def test_api_key_is_not_forwarded_to_shell(self) -> None:
        tool = create_bash_tool(self.cwd)
        with patch.dict(os.environ, {"PI_API_KEY": "do-not-leak"}):
            result = tool.execute(
                {"command": 'printf "${PI_API_KEY-unset}"'}, ToolContext(self.cwd)
            )
        self.assertEqual(result.output, "unset")

    def test_background_process_does_not_hold_tool_open(self) -> None:
        tool = create_bash_tool(self.cwd)
        started = time.monotonic()
        result = tool.execute(
            {"command": "sleep 5 & printf done"}, ToolContext(self.cwd)
        )
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(result.output, "done")


if __name__ == "__main__":
    unittest.main()
