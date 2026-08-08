import unittest

import numpy as np

from scripts.check_turbo_trace import check, check_frozen_proposals, parse_rows


TRACE = """
ROW 0 1 0.1 0.2 0.5 0.5 0.8 0 0 0
ROW 1 1 0.2 0.3 0.4 0.4 0.8 0 0 0
ROW 2 1 0.3 0.4 0.3 0.3 0.8 0 0 0
ROW 3 1 0.4 0.5 0.2 0.2 0.8 1 0 0
ROW 4 1 0.5 0.6 0.21 0.2 0.8 0 1 0
ROW 5 1 0.6 0.7 0.1 0.1 0.8 1 0 0
"""


class TurboTraceTests(unittest.TestCase):
    def test_independent_trust_replay_matches_trace(self):
        rows = parse_rows(TRACE, dimension=2)
        self.assertEqual(len(rows), 6)
        check(TRACE, dimension=2, initial=3)

    def test_changed_counter_is_rejected(self):
        changed = TRACE.replace("0.21 0.2 0.8 0 1 0", "0.21 0.2 0.8 1 0 0")
        with self.assertRaises(AssertionError):
            check(changed, dimension=2, initial=3)

    def test_frozen_initial_and_candidate_coordinates_are_checked(self):
        rows = parse_rows(TRACE, dimension=2)
        initial = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        candidates = np.array([
            [0.4, 0.5], [0.5, 0.6], [0.6, 0.7],
        ])
        check_frozen_proposals(rows, initial=3,
                               frozen_initial=initial,
                               frozen_candidates=candidates)

        changed = TRACE.replace("0.6 0.7 0.1", "0.61 0.7 0.1")
        with self.assertRaises(AssertionError):
            check_frozen_proposals(parse_rows(changed, dimension=2), initial=3,
                                   frozen_initial=initial,
                                   frozen_candidates=candidates)


if __name__ == "__main__":
    unittest.main()
