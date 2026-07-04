from nyra_ui.protocol import apply_event
from nyra_ui.state import HermesTaskView, UIPhase, UIState


def test_listening_keeps_llm_text_while_hermes_task_active():
    state = UIState(
        phase=UIPhase.SPEAKING,
        updated_at=0.0,
        llm_text="Queued that for Hermes as task-1.",
        is_final_llm=True,
        hermes_tasks=(
            HermesTaskView(label="task-1", status="running", elapsed_seconds=3.0),
        ),
    )
    next_state = state.with_phase(UIPhase.LISTENING)
    assert next_state.llm_text == "Queued that for Hermes as task-1."


def test_listening_keeps_llm_text_after_speaking():
    state = UIState(
        phase=UIPhase.SPEAKING,
        updated_at=0.0,
        llm_text="All done.",
        is_final_llm=True,
    )
    next_state = state.with_phase(UIPhase.LISTENING)
    assert next_state.llm_text == "All done."


def test_hermes_tasks_event_does_not_change_phase():
    state = UIState(phase=UIPhase.LISTENING, updated_at=0.0, llm_text="Working on it.")
    next_state = apply_event(
        state,
        {
            "type": "hermes_tasks",
            "tasks": [{"label": "task-1", "status": "queued", "elapsed_seconds": 0.0}],
        },
    )
    assert next_state.phase is UIPhase.LISTENING
    assert next_state.llm_text == "Working on it."
    assert len(next_state.hermes_tasks) == 1
