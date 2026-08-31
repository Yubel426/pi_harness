from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from pi_harness import Agent, AgentCallbacks, AgentError, Tool, ToolContext


def response(*, output, text="", response_id="resp_1", usage=None, status="completed"):
    return SimpleNamespace(
        id=response_id,
        status=status,
        model="gpt-5.6-sol",
        output=output,
        output_text=text,
        usage=usage,
    )


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **payload):
        self.requests.append(copy.deepcopy(payload))
        current = self.responses.pop(0)
        if isinstance(current, BaseException):
            raise current
        if payload.get("stream"):
            events = []
            if current.output_text:
                events.append(
                    SimpleNamespace(
                        type="response.output_text.delta", delta=current.output_text
                    )
                )
            events.append(SimpleNamespace(type="response.completed", response=current))
            return iter(events)
        return current


class AgentLoopTests(unittest.TestCase):
    def test_tool_loop_replays_reasoning_call_and_output(self) -> None:
        seen = []

        def execute(arguments, context: ToolContext):
            seen.append((dict(arguments), context.cwd))
            return "tool says 42"

        tool = Tool(
            name="answer",
            description="Return the answer",
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
            executor=execute,
        )
        first = response(
            output=[
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "answer",
                    "arguments": '{"question":"life"}',
                    "status": "completed",
                },
            ],
            response_id="resp_1",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
                output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )
        second = response(
            output=[
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "done", "annotations": []}
                    ],
                    "status": "completed",
                }
            ],
            text="done",
            response_id="resp_2",
        )
        fake = FakeResponses([first, second])
        deltas = []
        client = SimpleNamespace(responses=fake)
        agent = Agent(
            client=client,
            model="gpt-5.6-sol",
            instructions="test",
            tools=[tool],
            cwd=Path.cwd(),
            reasoning_effort="medium",
        )

        result = agent.run("find it", AgentCallbacks(on_text_delta=deltas.append))

        self.assertEqual(result.text, "done")
        self.assertEqual(result.api_rounds, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.usage.total_tokens, 14)
        self.assertEqual(result.usage.cached_tokens, 2)
        self.assertEqual(result.usage.reasoning_tokens, 3)
        self.assertEqual(deltas, ["done"])
        self.assertEqual(seen, [({"question": "life"}, Path.cwd().resolve())])

        second_input = fake.requests[1]["input"]
        self.assertEqual(second_input[0], {"role": "user", "content": "find it"})
        self.assertEqual(second_input[1]["type"], "reasoning")
        self.assertEqual(second_input[1]["encrypted_content"], "opaque")
        self.assertEqual(second_input[2]["type"], "function_call")
        self.assertEqual(
            second_input[3],
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "tool says 42",
            },
        )
        self.assertFalse(fake.requests[0]["store"])
        self.assertEqual(fake.requests[0]["reasoning"], {"effort": "medium"})
        self.assertEqual(fake.requests[0]["include"], ["reasoning.encrypted_content"])

    def test_invalid_arguments_become_tool_output(self) -> None:
        tool = Tool("noop", "noop", {"type": "object"}, lambda _args, _ctx: "unused")
        first = response(
            output=[
                {
                    "type": "function_call",
                    "call_id": "bad_call",
                    "name": "noop",
                    "arguments": "{broken",
                }
            ]
        )
        second = response(output=[], text="recovered")
        fake = FakeResponses([first, second])
        agent = Agent(
            client=SimpleNamespace(responses=fake),
            model="gpt-5.6-sol",
            instructions="test",
            tools=[tool],
            cwd=Path.cwd(),
            stream=False,
        )
        result = agent.run("go")
        self.assertEqual(result.text, "recovered")
        self.assertIn("invalid arguments", fake.requests[1]["input"][-1]["output"])

    def test_independent_runs_do_not_replay_previous_task(self) -> None:
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                fake = FakeResponses(
                    [
                        response(
                            output=[
                                {
                                    "type": "function_call",
                                    "call_id": "first_task_call",
                                    "name": "noop",
                                    "arguments": "{}",
                                }
                            ],
                            usage=SimpleNamespace(total_tokens=7),
                        ),
                        response(
                            output=[
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": "one"}],
                                }
                            ],
                            text="one",
                        ),
                        response(output=[], text="two"),
                    ]
                )
                agent = Agent(
                    client=SimpleNamespace(responses=fake),
                    model="gpt-5.6-sol",
                    instructions="test",
                    tools=[
                        Tool("noop", "test", {"type": "object"}, lambda _a, _c: "ok")
                    ],
                    cwd=Path.cwd(),
                    stream=streaming,
                )

                first_result = agent.run("first")
                second_result = agent.run("second")

                self.assertEqual(first_result.tool_calls, 1)
                self.assertEqual(first_result.usage.total_tokens, 7)
                self.assertEqual(second_result.text, "two")
                self.assertEqual(second_result.tool_calls, 0)
                self.assertEqual(second_result.usage.total_tokens, 0)
                self.assertEqual(
                    fake.requests[2]["input"], [{"role": "user", "content": "second"}]
                )

    def test_failed_or_interrupted_task_does_not_leave_context_for_next_run(
        self,
    ) -> None:
        failed = response(output=[], status="incomplete")
        failed.incomplete_details = SimpleNamespace(reason="max_output_tokens")
        for failure, expected_error in (
            (failed, AgentError),
            (KeyboardInterrupt(), KeyboardInterrupt),
        ):
            with self.subTest(error=expected_error.__name__):
                fake = FakeResponses(
                    [
                        response(
                            output=[
                                {
                                    "type": "function_call",
                                    "call_id": "old_task_call",
                                    "name": "noop",
                                    "arguments": "{}",
                                }
                            ]
                        ),
                        failure,
                        response(output=[], text="fresh start"),
                    ]
                )
                agent = Agent(
                    client=SimpleNamespace(responses=fake),
                    model="gpt-5.6-sol",
                    instructions="test",
                    tools=[
                        Tool("noop", "test", {"type": "object"}, lambda _a, _c: "ok")
                    ],
                    cwd=Path.cwd(),
                    stream=False,
                )
                with self.assertRaises(expected_error):
                    agent.run("failed task")
                result = agent.run("new task")
                self.assertEqual(result.text, "fresh start")
                self.assertEqual(
                    fake.requests[2]["input"], [{"role": "user", "content": "new task"}]
                )

    def test_single_prompt_can_run_multiple_tool_rounds(self) -> None:
        fake = FakeResponses(
            [
                response(
                    output=[
                        {
                            "type": "function_call",
                            "call_id": f"call_{index}",
                            "name": "step",
                            "arguments": f'{{"index":{index}}}',
                        }
                    ]
                )
                for index in (1, 2)
            ]
            + [response(output=[], text="done")]
        )
        executed = []

        def execute(arguments, _context):
            executed.append(arguments["index"])
            return f"result {arguments['index']}"

        agent = Agent(
            client=SimpleNamespace(responses=fake),
            model="gpt-5.6-sol",
            instructions="test",
            tools=[Tool("step", "test", {"type": "object"}, execute)],
            cwd=Path.cwd(),
            stream=False,
        )

        result = agent.run("finish both steps")

        self.assertEqual(executed, [1, 2])
        self.assertEqual(result.api_rounds, 3)
        self.assertEqual(result.tool_calls, 2)
        final_input = fake.requests[-1]["input"]
        self.assertEqual(sum(item.get("role") == "user" for item in final_input), 1)
        self.assertEqual(
            [
                item["output"]
                for item in final_input
                if item.get("type") == "function_call_output"
            ],
            ["result 1", "result 2"],
        )

    def test_incomplete_tool_call_is_not_executed(self) -> None:
        executed = False

        def dangerous(_arguments, _context):
            nonlocal executed
            executed = True
            return "should not run"

        tool = Tool("dangerous", "test", {"type": "object"}, dangerous)
        incomplete = response(
            status="incomplete",
            output=[
                {
                    "type": "function_call",
                    "call_id": "truncated_call",
                    "name": "dangerous",
                    "arguments": "{}",
                }
            ],
        )
        incomplete.incomplete_details = SimpleNamespace(reason="max_output_tokens")
        final = response(output=[], text="retried safely")
        fake = FakeResponses([incomplete, final])
        agent = Agent(
            client=SimpleNamespace(responses=fake),
            model="gpt-5.6-sol",
            instructions="test",
            tools=[tool],
            cwd=Path.cwd(),
            stream=False,
        )

        result = agent.run("go")

        self.assertFalse(executed)
        self.assertEqual(result.text, "retried safely")
        self.assertIn(
            "arguments may be truncated", fake.requests[1]["input"][-1]["output"]
        )


if __name__ == "__main__":
    unittest.main()
