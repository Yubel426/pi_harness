"""Offline tests against the installed upstream Pi runtime and a local HTTP stub."""

from __future__ import annotations

import json
import os
import pty
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from pi_harness import Agent, PiClient, PiError, Tool
from pi_harness.llm import runtime_directory


def openai_events(text="hello", *, tool=False):
    if tool:
        item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"value":42}',
            "status": "completed",
        }
    else:
        item = {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    response = {
        "id": "resp_local",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [item],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    events = [
        {"type": "response.created", "response": {"id": "resp_local"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": "", "content": []},
        },
    ]
    if not tool:
        events.append(
            {"type": "response.output_text.delta", "output_index": 0, "delta": text}
        )
    events.extend(
        [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
    )
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )


class StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload))
        body = self.server.responses.pop(0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body.encode())


@unittest.skipUnless(
    (runtime_directory() / "node_modules/@earendil-works/pi-ai/dist/index.js").exists(),
    "run pi-harness setup to enable Pi integration tests",
)
class BridgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.auth_file = Path(self.directory.name) / "auth.json"
        self.environment = patch.dict(
            os.environ,
            {
                key: ""
                for key in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                )
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def client(self, **kwargs):
        client = PiClient(auth_file=self.auth_file, timeout=15, **kwargs)
        self.addCleanup(client.close)
        return client

    def server(self, responses):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
        server.responses, server.requests = list(responses), []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join()

        self.addCleanup(close)
        return server, f"http://127.0.0.1:{server.server_port}/v1"

    def test_all_providers_and_oauth_methods_are_exposed(self):
        providers = self.client().get_providers()
        self.assertEqual(len(providers), 40)
        self.assertEqual(
            {p["id"] for p in providers if "oauth" in p["authMethods"]},
            {
                "openai-codex",
                "anthropic",
                "github-copilot",
                "openrouter",
                "xai",
                "kimi-coding",
                "radius",
            },
        )
        self.assertFalse(self.auth_file.exists())
        model = self.client().get_model("openai", "gpt-5.6-sol")
        self.assertIn("image", model["input"])

    def test_noninteractive_api_key_login_status_and_logout(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pi_harness",
                "login",
                "openai",
                "--api-key-stdin",
                "--auth-file",
                str(self.auth_file),
            ],
            input="test-only-secret\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("test-only-secret", result.stdout + result.stderr)
        status = self.client().auth_status("openai")
        self.assertTrue(status[0]["configured"])
        self.assertNotIn("test-only-secret", json.dumps(status))
        self.assertEqual(self.auth_file.stat().st_mode & 0o777, 0o600)
        self.client().logout("openai")
        self.assertEqual(json.loads(self.auth_file.read_text()), {})

    def test_oauth_only_provider_rejects_api_keys(self):
        with self.assertRaisesRegex(PiError, "does not accept API keys"):
            self.client().set_api_key("openai-codex", "test-only-secret")
        self.assertFalse(self.auth_file.exists())

    def test_interactive_api_key_login_masks_terminal_input(self):
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [
                "node",
                str(runtime_directory() / "bridge.mjs"),
                str(self.auth_file),
                "login",
                "openai",
                "api_key",
            ],
            stdin=slave,
            stdout=slave,
            stderr=slave,
        )
        os.close(slave)
        output = b""
        sent = False
        deadline = time.monotonic() + 10
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    output += chunk
                    if b"key" in output.lower() and not sent:
                        os.write(master, b"interactive-test-secret\n")
                        sent = True
                if process.poll() is not None:
                    break
            self.assertEqual(
                process.wait(timeout=2), 0, output.decode(errors="replace")
            )
            self.assertTrue(sent)
            self.assertNotIn(b"interactive-test-secret", output)
            self.assertEqual(
                json.loads(self.auth_file.read_text())["openai"]["key"],
                "interactive-test-secret",
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)

    def test_stored_oauth_cannot_be_redirected_to_custom_endpoint(self):
        self.auth_file.write_text(
            json.dumps(
                {
                    "openai-codex": {
                        "type": "oauth",
                        "access": "test",
                        "refresh": "test",
                        "expires": 9999999999999,
                    }
                }
            )
        )
        with self.assertRaisesRegex(
            PiError, "Custom endpoints cannot use stored OAuth"
        ):
            self.client(
                provider="openai-codex", base_url="https://example.test"
            ).complete("gpt-5.6-sol", {"messages": []})

    def test_unknown_model_fails_without_network(self):
        with self.assertRaisesRegex(PiError, "Unknown model"):
            self.client().get_model("openai", "not-a-model")

    def test_raw_complete_and_stream_use_real_pi_adapter(self):
        server, base_url = self.server([openai_events(), openai_events("world")])
        client = self.client(base_url=base_url, api_key="test-key")
        context = {"messages": [{"role": "user", "content": "hello", "timestamp": 1}]}
        message = client.complete("gpt-4o-mini", context)
        self.assertEqual(message["content"][0]["text"], "hello")
        self.assertEqual(message["provider"], "openai")
        events = list(client.stream_simple("gpt-4o-mini", context))
        self.assertEqual(
            "".join(e["delta"] for e in events if e["type"] == "text_delta"), "world"
        )
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(all(path == "/v1/responses" for path, _ in server.requests))
        self.assertFalse(server.requests[0][1]["store"])

    def test_real_pi_tool_round_trip_remains_single_prompt(self):
        server, base_url = self.server(
            [openai_events(tool=True), openai_events("done"), openai_events("fresh")]
        )
        seen = []
        tool = Tool(
            "echo",
            "Echo",
            {"type": "object", "properties": {"value": {"type": "integer"}}},
            lambda args, _ctx: seen.append(args["value"]) or "42",
        )
        agent = Agent(
            client=self.client(base_url=base_url, api_key="test-key"),
            model="gpt-4o-mini",
            instructions="test",
            tools=[tool],
            cwd=Path.cwd(),
        )
        result = agent.run("go")
        self.assertEqual(
            (result.text, result.tool_calls, result.api_rounds), ("done", 1, 2)
        )
        self.assertEqual(seen, [42])
        self.assertTrue(
            any(
                item.get("type") == "function_call_output"
                for item in server.requests[1][1]["input"]
            )
        )
        self.assertEqual(agent.run("new task").text, "fresh")
        self.assertFalse(
            any(
                item.get("type") == "function_call_output"
                for item in server.requests[2][1]["input"]
            )
        )

    def test_anthropic_and_chat_completions_share_the_pi_message_schema(self):
        anthropic = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        anthropic_body = "".join(
            f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in anthropic
        )
        completions = [
            {
                "id": "chatcmpl_local",
                "object": "chat.completion.chunk",
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_local",
                "object": "chat.completion.chunk",
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        ]
        completion_body = (
            "".join(f"data: {json.dumps(e)}\n\n" for e in completions)
            + "data: [DONE]\n\n"
        )
        for provider, body in (
            ("anthropic", anthropic_body),
            ("groq", completion_body),
        ):
            with self.subTest(provider=provider):
                server, url = self.server([body])
                client = self.client(
                    provider=provider,
                    base_url=url.removesuffix("/v1")
                    if provider == "anthropic"
                    else url,
                    api_key="test-key",
                )
                model = client.get_models(provider)[0]["id"]
                message = client.complete_simple(
                    model,
                    {
                        "messages": [
                            {"role": "user", "content": "hello", "timestamp": 1}
                        ]
                    },
                )
                self.assertEqual(message["provider"], provider)
                self.assertEqual(
                    message["stopReason"], "stop", message.get("errorMessage")
                )
                self.assertEqual(message["content"][0]["text"], "hello")
                self.assertEqual(len(server.requests), 1)


if __name__ == "__main__":
    unittest.main()
