from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pi_harness.config import HarnessConfig, normalize_base_url


class ConfigTests(unittest.TestCase):
    def test_normalize_base_url_adds_v1(self) -> None:
        self.assertEqual(
            normalize_base_url("https://api.example.test/"),
            "https://api.example.test/v1",
        )
        self.assertEqual(
            normalize_base_url("https://api.example.test/openai/v1"),
            "https://api.example.test/openai/v1",
        )

    def test_environment_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PI_API_KEY": "pi-secret",
                "OPENAI_API_KEY": "openai-secret",
                "PI_BASE_URL": "https://provider.example",
                "PI_MODEL": "gpt-5.6-sol",
            },
            clear=True,
        ):
            config = HarnessConfig.from_environment(cwd=Path.cwd())
        self.assertEqual(config.api_key, "pi-secret")
        self.assertEqual(config.base_url, "https://provider.example/v1")
        self.assertEqual(config.model, "gpt-5.6-sol")

    def test_credentials_are_resolved_by_pi(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = HarnessConfig.from_environment(provider="openai-codex")
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.base_url)
        self.assertEqual(config.provider, "openai-codex")

    def test_openai_environment_does_not_leak_to_other_providers(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "https://example.test"},
            clear=True,
        ):
            config = HarnessConfig.from_environment(
                provider="anthropic", base_url="https://api.anthropic.com"
            )
        self.assertIsNone(config.api_key)
        self.assertEqual(config.base_url, "https://api.anthropic.com")

    def test_explicit_off_disables_reasoning_parameter(self) -> None:
        config = HarnessConfig.from_environment(
            api_key="test-key", reasoning_effort="off", cwd=Path.cwd()
        )
        self.assertIsNone(config.reasoning_effort)


if __name__ == "__main__":
    unittest.main()
