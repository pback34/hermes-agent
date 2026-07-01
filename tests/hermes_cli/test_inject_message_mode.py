"""P2c-delivery A3 — inject_message(mode=) routing (queue vs interrupt).

The poller (external hermes-collab plugin) passes a delivery mode so a real-time
message can reach a live session WITHOUT preempting its current turn:
  - 'interrupt' (default, legacy) — preempt a busy agent (goes to _interrupt_queue).
  - 'queue' — non-interrupting; a busy agent picks it up at the next turn boundary.
When the agent is idle there is nothing to interrupt, so every mode queues as the
next input. Pure unit test with a fake CLI/manager (no live agent, no DB).
"""
import queue

from hermes_cli.plugins import PluginContext


class _FakeCLI:
    def __init__(self, running: bool):
        self._agent_running = running
        self._interrupt_queue = queue.Queue()
        self._pending_input = queue.Queue()


class _FakeManager:
    def __init__(self, cli):
        self._cli_ref = cli


def _ctx(cli):
    return PluginContext(None, _FakeManager(cli))


def test_queue_mode_does_not_interrupt_a_busy_agent():
    cli = _FakeCLI(running=True)
    assert _ctx(cli).inject_message("hi", mode="queue") is True
    assert cli._interrupt_queue.qsize() == 0            # NOT preempted
    assert cli._pending_input.get_nowait() == "hi"      # waits for next turn


def test_interrupt_mode_preempts_a_busy_agent():
    cli = _FakeCLI(running=True)
    _ctx(cli).inject_message("hi", mode="interrupt")
    assert cli._interrupt_queue.get_nowait() == "hi"
    assert cli._pending_input.qsize() == 0


def test_default_mode_preserves_legacy_interrupt_when_busy():
    cli = _FakeCLI(running=True)
    _ctx(cli).inject_message("hi")                       # no mode → legacy behavior
    assert cli._interrupt_queue.get_nowait() == "hi"


def test_idle_agent_always_queues_regardless_of_mode():
    cli = _FakeCLI(running=False)
    _ctx(cli).inject_message("hi", mode="interrupt")     # even interrupt mode
    assert cli._pending_input.get_nowait() == "hi"
    assert cli._interrupt_queue.qsize() == 0


def test_gateway_mode_no_cli_ref_returns_false():
    assert PluginContext(None, _FakeManager(None)).inject_message("hi", mode="queue") is False
