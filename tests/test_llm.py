from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pi_harness import Agent, AgentCallbacks, AgentError, PiClient, PiError, Tool


def assistant(content, *, stop="stop", provider="anthropic", **extra):
    return {
        "role": "assistant",
        "provider": provider,
        "api": "anthropic-messages",
        "model": "test-model",
        "content": content,
        "stopReason": stop,
        "timestamp": 1,
        "usage": {
            "input": 3,
            "output": 2,
            "cacheRead": 4,
            "cacheWrite": 1,
            "totalTokens": 10,
        },
        **extra,
    }


class FakePiClient(PiClient):
    def __init__(self, messages):
        super().__init__()
        self.messages = iter(messages)
        self.requests = []

    def complete_simple(self, model, context, options=None):
        self.requests.append(copy.deepcopy((model, context, options)))
        value = next(self.messages)
        if isinstance(value, BaseException):
            raise value
        return value

    def stream_simple(self, model, context, options=None):
        value = self.complete_simple(model, context, options)
        for item in value["content"]:
            if item["type"] == "text":
                yield {"type": "text_delta", "delta": item["text"], "contentIndex": 0}
        yield {"type": "done", "message": value, "reason": value["stopReason"]}


class PiAdapterTests(unittest.TestCase):
    def make_agent(self, client, *, stream=True, tools=()):
        return Agent(
            client=client,
            model="test-model",
            instructions="test",
            tools=tools,
            cwd=Path.cwd(),
            stream=stream,
        )

    def test_preserves_native_messages_signatures_and_multiple_calls(self):
        first = assistant(
            [
                {
                    "type": "thinking",
                    "thinking": "reason",
                    "thinkingSignature": "opaque-signature",
                },
                {
                    "type": "toolCall",
                    "id": "one",
                    "name": "echo",
                    "arguments": {"n": 1},
                },
                {"type": "toolCall", "id": "two", "name": "missing", "arguments": {}},
            ],
            stop="toolUse",
            responseId="msg-1",
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                client = FakePiClient(
                    [first, assistant([{"type": "text", "text": "done"}])]
                )
                tool = Tool("echo", "echo", {"type": "object"}, lambda args, _ctx: args)
                agent = self.make_agent(client, stream=stream, tools=[tool])
                deltas = []
                result = agent.run("go", AgentCallbacks(on_text_delta=deltas.append))
                self.assertEqual(result.text, "done")
                self.assertEqual(result.tool_calls, 2)
                self.assertEqual(result.usage.total_tokens, 20)
                self.assertEqual(result.usage.input_tokens, 16)
                self.assertEqual(deltas, ["done"])
                context = client.requests[1][1]
                self.assertEqual(context["messages"][1], first)
                self.assertEqual(context["messages"][2]["toolCallId"], "one")
                self.assertFalse(context["messages"][2]["isError"])
                self.assertTrue(context["messages"][3]["isError"])
                self.assertEqual(context["tools"][0]["name"], "echo")
                self.assertEqual(client.requests[0][2]["reasoning"], "medium")

    def test_no_history_between_tasks_after_success_or_failure(self):
        for failure in (None, PiError("failed"), KeyboardInterrupt()):
            messages = [
                failure or assistant([{"type": "text", "text": "first"}]),
                assistant([{"type": "text", "text": "second"}]),
            ]
            client = FakePiClient(messages)
            agent = self.make_agent(client)
            if failure:
                with self.assertRaises(type(failure)):
                    agent.run("first")
            else:
                agent.run("first")
            self.assertEqual(agent.run("second").text, "second")
            self.assertEqual(len(client.requests[-1][1]["messages"]), 1)
            self.assertEqual(client.requests[-1][1]["messages"][0]["content"], "second")

    def test_length_stop_never_executes_partial_tools(self):
        client = FakePiClient(
            [
                assistant(
                    [
                        {
                            "type": "toolCall",
                            "id": "bad",
                            "name": "side_effect",
                            "arguments": {},
                        }
                    ],
                    stop="length",
                ),
                assistant([{"type": "text", "text": "recovered"}]),
            ]
        )
        calls = []
        tool = Tool("side_effect", "test", {}, lambda *_: calls.append(1))
        self.assertEqual(
            self.make_agent(client, tools=[tool]).run("go").text, "recovered"
        )
        self.assertEqual(calls, [])
        self.assertTrue(client.requests[1][1]["messages"][-1]["isError"])

    def test_provider_error_does_not_execute_tools(self):
        client = FakePiClient(
            [assistant([], stop="error", errorMessage="provider failed")]
        )
        with self.assertRaisesRegex(PiError, "provider failed"):
            self.make_agent(client).run("go")

    def test_max_tool_rounds_remains_bounded(self):
        message = assistant(
            [{"type": "toolCall", "id": "one", "name": "missing", "arguments": {}}],
            stop="toolUse",
        )
        client = FakePiClient([message, message])
        agent = self.make_agent(client)
        agent.max_tool_rounds = 1
        with self.assertRaises(AgentError):
            agent.run("go")

    def test_missing_runtime_does_not_auto_install(self):
        with tempfile.TemporaryDirectory() as directory:
            client = PiClient(runtime_dir=directory)
            with self.assertRaisesRegex(PiError, "setup"):
                client.get_providers()

    def test_complete_passes_pi_json_without_normalization(self):
        context = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "data": "base64", "mimeType": "image/png"}
                    ],
                    "timestamp": 1,
                }
            ]
        }
        client = PiClient(provider="google")
        with patch.object(client, "_call", return_value={"native": True}) as call:
            self.assertEqual(
                client.complete("gemini", context, {"temperature": 0.2}),
                {"native": True},
            )
        self.assertEqual(call.call_args.kwargs["context"], context)
        self.assertEqual(call.call_args.kwargs["options"], {"temperature": 0.2})

    def test_timeout_terminates_bridge_without_leaving_processes(self):
        client = PiClient(timeout=0.1)
        started = time.monotonic()
        with (
            patch.object(
                client,
                "_command",
                return_value=[sys.executable, "-c", "import time; time.sleep(20)"],
            ),
            self.assertRaisesRegex(PiError, "timed out"),
        ):
            client.get_providers()
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(client._processes)

    def test_client_close_cancels_an_abandoned_stream(self):
        frame = json.dumps({"event": {"type": "start"}})
        script = f"import time; print({frame!r}, flush=True); time.sleep(20)"
        client = PiClient(timeout=10)
        with patch.object(
            client, "_command", return_value=[sys.executable, "-c", script]
        ):
            events = client.stream("test", {"messages": []})
            self.assertEqual(next(events)["type"], "start")
            client.close()
        self.assertFalse(client._processes)
        self.assertFalse(client._streams)
        self.assertEqual(list(events), [])


if __name__ == "__main__":
    unittest.main()
