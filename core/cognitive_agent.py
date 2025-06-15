"""
Stub CognitiveAgent for ARC pipeline—
delegates to your rule-based solvers for each category.
"""

import numpy as np
from arc_prize_solvers import (
    solve_identity,
    solve_uniform_mapping,
    solve_non_uniform,
)

class CognitiveAgent:
    def __init__(self, agent_id=None):
        self.agent_id = agent_id or "stub-agent"
        self._challenge = None
        self._last_prediction = None
        self._last_reward = None

    def load_challenge(self, challenge: dict):
        """Store the ARC challenge dict."""
        self._challenge = challenge

    def run_episode(self):
        """Produce a grid→grid prediction using rule-based solvers."""
        if self._challenge is None:
            raise RuntimeError("Call load_challenge() first.")

        ch = self._challenge
        inp = np.array(ch['train'][0]['input'], dtype=int)

        # Determine category by comparing input/output shape
        out_list = ch['train'][0].get('output')
        if out_list is None:
            # no train-output, echo
            self._last_prediction = inp.copy()
            return

        tgt = np.array(out_list, dtype=int)
        h0, w0 = inp.shape
        h1, w1 = tgt.shape

        if (h0, w0) == (h1, w1):
            # identity
            pred = solve_identity(ch, ch)
        elif h1 % h0 == 0 and w1 % w0 == 0:
            # uniform scale
            pred = solve_uniform_mapping(ch, ch)
        else:
            # non-uniform
            pred = solve_non_uniform(ch, ch)

        self._last_prediction = pred

    def get_output(self):
        """Return last predicted grid as list-of-lists."""
        if self._last_prediction is None:
            raise RuntimeError("Call run_episode() first.")
        return self._last_prediction.tolist()

    def learn(self, reward: float):
        """Record the reward (no-op)."""
        self._last_reward = reward

    async def process_input(self, text: str) -> str:
        return text

    def get_agent_status(self):
        return {"agent_id": self.agent_id, "last_reward": self._last_reward}
