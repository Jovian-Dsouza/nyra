import unittest

from orchestrator.state import State, StateMachine


class StateMachineTests(unittest.TestCase):
    def test_happy_path_transitions(self) -> None:
        sm = StateMachine()
        sm.transition(State.LISTENING, "wake")
        sm.transition(State.THINKING, "final")
        sm.transition(State.SPEAKING, "first-sentence")
        sm.transition(State.IDLE, "done")
        self.assertEqual(sm.state, State.IDLE)

    def test_invalid_transition_raises(self) -> None:
        sm = StateMachine()
        with self.assertRaises(ValueError):
            sm.transition(State.SPEAKING, "invalid")

    def test_barge_in_forces_listening(self) -> None:
        sm = StateMachine(state=State.SPEAKING)
        sm.set_listening_for_barge_in()
        self.assertEqual(sm.state, State.LISTENING)


if __name__ == "__main__":
    unittest.main()

